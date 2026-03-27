"""
FlowEdit adapted for FLUX.2-dev.

Core algorithm identical to FlowEdit_utils.py FlowEditFLUX,
but with FLUX.2 pipeline interfaces (different VAE, text encoder, transformer).
"""

import numpy as np
import torch
from tqdm import tqdm
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps


def calc_v_flux2(pipe, latents, prompt_embeds, guidance, text_ids, latent_image_ids, t):
    """Compute velocity prediction with FLUX.2 transformer."""
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

    return noise_pred


@torch.no_grad()
def FlowEditFLUX2(
    pipe,
    x_src,
    src_prompt,
    tar_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 15,
):
    """
    FlowEdit for FLUX.2-dev.

    Args:
        pipe: Flux2Pipeline (loaded with device_map)
        x_src: source latents after VAE encode + patchify + batch norm (B, C, H, W)
        src_prompt: source prompt string
        tar_prompt: target prompt string
        T_steps: number of ODE steps
        n_avg: number of averaging iterations per step
        src_guidance_scale: guidance for source velocity
        tar_guidance_scale: guidance for target velocity
        n_min: pure generation steps at end (0 = all editing)
        n_max: editing steps from the end

    Returns:
        edited latents (packed, same shape as input to _pack_latents output)
    """
    device = x_src.device
    dtype = x_src.dtype

    # Prepare latent IDs
    latent_ids = pipe._prepare_latent_ids(x_src)
    latent_ids = latent_ids.to(device)

    # Pack source latents
    x_src_packed = pipe._pack_latents(x_src)
    image_seq_len = x_src_packed.shape[1]

    # Prepare timesteps with dynamic shifting
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=T_steps)
    timesteps, T_steps = retrieve_timesteps(
        pipe.scheduler,
        T_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    # Encode prompts
    src_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt,
        device=pipe.transformer.device,
        num_images_per_prompt=1,
    )

    tar_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt,
        device=pipe.transformer.device,
        num_images_per_prompt=1,
    )

    # Guidance embeddings
    src_guidance = torch.tensor([src_guidance_scale], device=pipe.transformer.device, dtype=torch.float32)
    src_guidance = src_guidance.expand(x_src_packed.shape[0])
    tar_guidance = torch.tensor([tar_guidance_scale], device=pipe.transformer.device, dtype=torch.float32)
    tar_guidance = tar_guidance.expand(x_src_packed.shape[0])

    # Move everything to a single compute device
    tdev = device  # use the device where latents already are
    tdtype = torch.bfloat16
    x_src_packed = x_src_packed.to(tdev, dtype=tdtype)
    latent_ids = latent_ids.to(tdev, dtype=tdtype)
    src_prompt_embeds = src_prompt_embeds.to(tdev, dtype=tdtype)
    tar_prompt_embeds = tar_prompt_embeds.to(tdev, dtype=tdtype)
    src_text_ids = src_text_ids.to(tdev, dtype=tdtype)
    tar_text_ids = tar_text_ids.to(tdev, dtype=tdtype)
    src_guidance = src_guidance.to(tdev)
    tar_guidance = tar_guidance.to(tdev)

    # Initialize edited latents
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit"):
        if T_steps - i > n_max:
            continue

        pipe.scheduler._init_step_index(t)
        t_i = pipe.scheduler.sigmas[pipe.scheduler.step_index]
        if i < len(timesteps):
            t_im1 = pipe.scheduler.sigmas[pipe.scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:
            # FlowEdit: compute V_tar - V_src and apply
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed)

                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = calc_v_flux2(
                    pipe, latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_ids,
                    t=t,
                )

                Vt_tar = calc_v_flux2(
                    pipe, latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_ids,
                    t=t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            step_size = (t_im1 - t_i)
            delta_norm = V_delta_avg.abs().mean().item()
            edit_norm = (step_size * V_delta_avg).abs().mean().item()
            print(f"  step {i}: t_i={t_i:.4f}, t_im1={t_im1:.4f}, step_size={step_size:.4f}, "
                  f"|V_delta|={delta_norm:.6f}, |edit|={edit_norm:.6f}")

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            # Regular sampling for last n_min steps (SDEdit-style)
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed)
                xt_src = pipe.scheduler.scale_noise(x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux2(
                pipe, latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_ids,
                t=t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar
    return out, latent_ids
