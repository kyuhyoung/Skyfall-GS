#!/bin/bash

# FlowEdit schnell extra grid search
# 8 additional TIFs needed for heatmaps fes_02, fes_05, fes_06, fes_09
# 7 jobs → 11 TIFs generated (no tg/sg for schnell)
#
# All workers share a single job queue with file locking.
# RAM-safe: measures peak RAM on first tile, auto-limits worker count.
#
# Usage:
#   ./grid_search_schnell_extra.sh
#   ./grid_search_schnell_extra.sh --gpus 1,2,3

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output/grid_schnell_extra_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$OUTPUT_DIR"

LOGFILE="${OUTPUT_DIR}/grid_search.log"
exec > >(stdbuf -oL tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse --gpus (empty = auto-detect each round)
USER_GPUS=""
for arg in "$@"; do
    if [ "$prev_arg" = "--gpus" ]; then
        USER_GPUS="$arg"
        prev_arg=""
        continue
    fi
    if [ "$arg" = "--gpus" ]; then
        prev_arg="$arg"
        continue
    fi
done

PYTHON="/media2/4tb/kevin/envs/skyfall/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python)"
fi

export HF_HOME=/media2/4tb/kevin/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

INPUT="/media2/data/dataset_stereo/satelite/korea/seoul/gangnam/samsung/fused_top_naive.tif"

MODEL="black-forest-labs/FLUX.1-schnell"
GAMMA=0.7
TILE_SIZE=1024
OVERLAP=128

SRC_PROMPT="Satellite image with black missing regions, noise, blurring, and low resolution"
TAR_PROMPT="Complete high resolution satellite image with all areas naturally filled with buildings, roads, and vegetation, sharp details and vivid colors"

JOB_FILE="${SCRIPT_DIR}/output/jobs_schnell_extra.json"

# --------------- Create shared job queue ---------------
QUEUE_FILE="${OUTPUT_DIR}/job_queue.json"
QUEUE_LOCK="${OUTPUT_DIR}/job_queue.lock"

$PYTHON -c "
import json
with open('${JOB_FILE}') as f:
    jobs = json.load(f)
for i, j in enumerate(jobs):
    j['job_id'] = i
    j['status'] = 'pending'
with open('${QUEUE_FILE}', 'w') as f:
    json.dump(jobs, f, indent=2)
"
touch "$QUEUE_LOCK"

TOTAL_JOBS=$($PYTHON -c "import json; print(len(json.load(open('${QUEUE_FILE}'))))")
TOTAL_TIFS=$($PYTHON -c "import json; print(sum(j.get('end_pass',2)-j.get('start_pass',1)+1 for j in json.load(open('${QUEUE_FILE}'))))")

echo ""
echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  FlowEdit Schnell Extra Grid Search${NC}"
echo -e "${CYAN}  8 TIFs for heatmaps fes_02, fes_05, fes_06, fes_09${NC}"
echo -e "${CYAN}  No tg/sg (schnell ignores guidance)${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Model: ${MODEL}"
echo "Candidate GPUs: ${USER_GPUS:-auto-detect}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Total: ${TOTAL_JOBS} jobs, ${TOTAL_TIFS} TIFs"
echo ""

# --------------- Helper: get RAM threshold (adaptive) ---------------
RAM_COST_FILE="${OUTPUT_DIR}/ram_per_model.txt"

get_ram_threshold() {
    # Returns min RAM needed: max(measured*1.2, measured+40GB reserve), or 50GB fallback
    $PYTHON -c "
import os
cost_file = '${RAM_COST_FILE}'
if os.path.exists(cost_file):
    measured = float(open(cost_file).read().strip())
    print(f'{max(measured * 1.2, measured + 40):.1f}')
else:
    print('50.0')
"
}

# --------------- Helper: detect free GPUs ---------------
detect_free_gpus() {
    local RAM_THRESH=$(get_ram_threshold)
    $PYTHON -c "
import subprocess, psutil

user_gpus = '${USER_GPUS}'
ram_thresh = ${RAM_THRESH}

# Get busy GPU UUIDs (running python)
r = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=gpu_uuid,process_name', '--format=csv,noheader'],
    capture_output=True, text=True)
busy = set()
for line in r.stdout.strip().split('\n'):
    if 'python' in line.lower():
        busy.add(line.split(',')[0].strip())

# Map index -> UUID
r2 = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader'],
    capture_output=True, text=True)
free = []
for line in r2.stdout.strip().split('\n'):
    idx, uuid = [x.strip() for x in line.split(',')]
    if uuid not in busy:
        if user_gpus:
            if idx in user_gpus.split(','):
                free.append(idx)
        else:
            free.append(idx)

# Check RAM
avail_gb = psutil.virtual_memory().available / (1024 ** 3)
if avail_gb < ram_thresh:
    print('')
else:
    print(','.join(free))
"
}

# --------------- Helper: real-time GPU+RAM check for one GPU ---------------
check_gpu_ready() {
    local GPU_ID=$1
    local RAM_THRESH=$(get_ram_threshold)
    $PYTHON -c "
import subprocess, psutil
gpu_id = ${GPU_ID}
ram_thresh = ${RAM_THRESH}
r = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name', '--format=csv,noheader'],
    capture_output=True, text=True)
r2 = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,gpu_bus_id', '--format=csv,noheader'],
    capture_output=True, text=True)
idx_to_bus = {}
for line in r2.stdout.strip().split('\n'):
    parts = [x.strip() for x in line.split(',')]
    if len(parts) == 2:
        idx_to_bus[int(parts[0])] = parts[1]
my_bus = idx_to_bus.get(gpu_id, '')
gpu_busy = False
for line in r.stdout.strip().split('\n'):
    if my_bus and my_bus in line and 'python' in line.lower():
        gpu_busy = True
        break
avail_gb = psutil.virtual_memory().available / (1024 ** 3)
if gpu_busy:
    print(f'GPU_BUSY {avail_gb:.1f} {ram_thresh:.1f}')
elif avail_gb < ram_thresh:
    print(f'RAM_LOW {avail_gb:.1f} {ram_thresh:.1f}')
else:
    print(f'OK {avail_gb:.1f} {ram_thresh:.1f}')
"
}

# --------------- Main retry loop ---------------
T_START=$(date +%s)
MAX_ROUNDS=10
WAIT_SEC=60
TOTAL_FAILED=0

for ((round=1; round<=MAX_ROUNDS; round++)); do

    # Reset crashed workers (in_progress → pending)
    if ((round > 1)); then
        $PYTHON -c "
import json
with open('${QUEUE_FILE}') as f:
    jobs = json.load(f)
reset = 0
for j in jobs:
    if j['status'] == 'in_progress':
        j['status'] = 'pending'
        reset += 1
if reset:
    with open('${QUEUE_FILE}', 'w') as f:
        json.dump(jobs, f, indent=2)
    print(f'Reset {reset} stuck jobs back to pending')
"
    fi

    # Count pending jobs
    PENDING=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status']=='pending'))")
    DONE=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status']=='done'))")

    if [ "$PENDING" = "0" ]; then
        echo -e "${GREEN}All jobs complete! (${DONE}/${TOTAL_JOBS} done)${NC}"
        break
    fi

    echo -e "${CYAN}=== Round ${round}: ${PENDING} pending, ${DONE} done ===${NC}"

    # Detect free GPUs (real-time)
    FREE_GPUS=$(detect_free_gpus)

    if [ -z "$FREE_GPUS" ]; then
        CUR_THRESH=$(get_ram_threshold)
        echo -e "${YELLOW}No free GPUs (or RAM < ${CUR_THRESH}GB). Waiting ${WAIT_SEC}s before retry...${NC}"
        sleep "$WAIT_SEC"
        continue
    fi

    echo -e "${GREEN}Free GPUs this round: ${FREE_GPUS}${NC}"
    IFS=',' read -ra FREE_IDS <<< "$FREE_GPUS"

    # Launch workers sequentially (wait for first tile peak RAM between each)
    PIDS=()
    LAUNCHED=()
    WORST_AVAIL=""  # will be set after first worker's first tile
    FIRST_PEAK=""   # first worker's clean peak (no interference)
    MODELS_LOADED=0  # count of additional model loads after first worker

    for GPU_ID in "${FREE_IDS[@]}"; do
        # Re-check this specific GPU right before launch (could have been taken since detect)
        PRE_CHECK=$(check_gpu_ready "$GPU_ID")
        CHECK_STATUS=$(echo "$PRE_CHECK" | awk '{print $1}')
        CHECK_RAM=$(echo "$PRE_CHECK" | awk '{print $2}')
        CHECK_THRESH=$(echo "$PRE_CHECK" | awk '{print $3}')

        if [ "$CHECK_STATUS" = "GPU_BUSY" ]; then
            echo -e "${RED}[SKIP] GPU ${GPU_ID}: python process detected since detection.${NC}"
            continue
        fi

        # For 2nd+ workers: use worst_available - (models loaded × first worker's clean peak)
        if [ -n "$WORST_AVAIL" ] && [ ${#LAUNCHED[@]} -ge 1 ]; then
            RAM_COST=$($PYTHON -c "
wa = ${WORST_AVAIL}
peak = ${FIRST_PEAK}
loaded = ${MODELS_LOADED}
effective = wa - (peak * loaded)
thresh = max(peak * 1.2, peak + 40)
print(f'{effective:.1f} {thresh:.1f}')
")
            EFF_AVAIL=$(echo "$RAM_COST" | awk '{print $1}')
            EFF_THRESH=$(echo "$RAM_COST" | awk '{print $2}')
            IS_LOW=$($PYTHON -c "print('yes' if ${EFF_AVAIL} < ${EFF_THRESH} else 'no')")
            if [ "$IS_LOW" = "yes" ]; then
                echo -e "${RED}[SKIP] GPU ${GPU_ID}: effective RAM too low (${EFF_AVAIL}GB < ${EFF_THRESH}GB needed). Stopping launches.${NC}"
                break
            fi
            echo -e "${CYAN}  Effective available: ${EFF_AVAIL}GB (worst ${WORST_AVAIL}GB - ${MODELS_LOADED} × ${FIRST_PEAK}GB)${NC}"
        elif [ "$CHECK_STATUS" = "RAM_LOW" ]; then
            echo -e "${RED}[SKIP] GPU ${GPU_ID}: RAM too low (${CHECK_RAM}GB < ${CHECK_THRESH}GB needed). Stopping launches for this round.${NC}"
            break
        fi

        # Check if there are still pending jobs (other workers may have taken them all)
        STILL_PENDING=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status']=='pending'))")
        if [ "$STILL_PENDING" = "0" ]; then
            echo -e "${GREEN}No more pending jobs. Skipping remaining GPUs.${NC}"
            break
        fi

        echo -e "${YELLOW}GPU ${GPU_ID}: OK (RAM ${CHECK_RAM}GB, no python) — launching worker${NC}"
        WORKER_LOG="${OUTPUT_DIR}/worker_gpu${GPU_ID}_r${round}.log"

        $PYTHON -u "${SCRIPT_DIR}/grid_worker.py" \
            --queue_file "$QUEUE_FILE" \
            --queue_lock "$QUEUE_LOCK" \
            --input "$INPUT" \
            --output_dir "$OUTPUT_DIR" \
            --tile_size "$TILE_SIZE" \
            --overlap "$OVERLAP" \
            --gamma "$GAMMA" \
            --max_pass 2 \
            --model "$MODEL" \
            --device "cuda:${GPU_ID}" \
            --src_prompt "$SRC_PROMPT" \
            --tar_prompt "$TAR_PROMPT" \
            > "$WORKER_LOG" 2>&1 &

        PIDS+=($!)
        LAUNCHED+=($GPU_ID)

        # Every worker: wait for first tile peak RAM measurement before launching next
        echo -e "${CYAN}  Waiting for GPU ${GPU_ID} first tile (peak RAM measurement)...${NC}"
        while true; do
            if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
                echo -e "${RED}  GPU ${GPU_ID} worker exited during first tile. Check ${WORKER_LOG}${NC}"
                break
            fi
            if grep -q "Peak RAM during first tile" "$WORKER_LOG" 2>/dev/null; then
                PEAK_LINE=$(grep "Peak RAM during first tile" "$WORKER_LOG")
                echo -e "${GREEN}  GPU ${GPU_ID} first tile done. ${PEAK_LINE}${NC}"
                if [ ${#LAUNCHED[@]} -eq 1 ]; then
                    # First worker: read worst-case available and clean peak as baseline
                    WORST_AVAIL_FILE="${OUTPUT_DIR}/worst_available.txt"
                    if [ -f "$WORST_AVAIL_FILE" ]; then
                        WORST_AVAIL=$(cat "$WORST_AVAIL_FILE")
                        echo -e "${CYAN}  Worst-case available with 1 worker at peak: ${WORST_AVAIL}GB${NC}"
                    fi
                    FIRST_PEAK=$(cat "${RAM_COST_FILE}")
                    echo -e "${CYAN}  First worker clean peak: ${FIRST_PEAK}GB (used for launch decisions)${NC}"
                else
                    # Only count workers AFTER the first one (first worker's cost is in worst_available)
                    MODELS_LOADED=$((MODELS_LOADED + 1))
                fi
                break
            fi
            sleep 2
        done
    done

    NUM_LAUNCHED=${#LAUNCHED[@]}
    if [ "$NUM_LAUNCHED" = "0" ]; then
        echo -e "${YELLOW}No workers launched this round. Waiting ${WAIT_SEC}s...${NC}"
        sleep "$WAIT_SEC"
        continue
    fi

    echo ""
    echo -e "${CYAN}${NUM_LAUNCHED} workers running (GPUs: ${LAUNCHED[*]}). Waiting for completion...${NC}"
    echo -e "${CYAN}Monitor: tail -f ${OUTPUT_DIR}/worker_gpu*_r${round}.log${NC}"
    echo ""

    # Wait for all workers in this round
    ROUND_FAILED=0
    for ((g=0; g<NUM_LAUNCHED; g++)); do
        GPU_ID=${LAUNCHED[$g]}
        if ! wait "${PIDS[$g]}"; then
            echo -e "${RED}[ERROR] Worker GPU ${GPU_ID} failed (round ${round}).${NC}"
            ROUND_FAILED=$((ROUND_FAILED + 1))
        else
            echo -e "${GREEN}Worker GPU ${GPU_ID} finished (round ${round}).${NC}"
        fi
    done
    TOTAL_FAILED=$((TOTAL_FAILED + ROUND_FAILED))
    echo ""
done

# --------------- Final summary ---------------
T_END=$(date +%s)
T_ELAPSED=$(( T_END - T_START ))

FINAL_DONE=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status']=='done'))")
FINAL_FAILED=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status']=='failed'))")
FINAL_PENDING=$($PYTHON -c "import json; print(sum(1 for j in json.load(open('${QUEUE_FILE}')) if j['status'] in ('pending','in_progress')))")

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}  Grid Search Complete${NC}"
echo -e "${CYAN}==============================================================${NC}"
echo "Total time: ${T_ELAPSED}s ($(echo "scale=1; ${T_ELAPSED}/60" | bc)min)"
echo "Jobs: ${FINAL_DONE} done, ${FINAL_FAILED} failed, ${FINAL_PENDING} remaining"
echo "TIFs: $(ls "${OUTPUT_DIR}"/*.tif 2>/dev/null | wc -l) / ${TOTAL_TIFS}"
echo "Queue: ${QUEUE_FILE}"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
