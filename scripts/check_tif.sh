#!/bin/bash
# 241 서버 용 tif
# TIF_PATH="${1:-/data/satellite/seoul/gangnam/samsung/gwarp_out_ps_ba/fused_top_naive.tif}"
TIF_PATH="${1:-/data/dataset/sat/korea/seoul/samsung/fused_top_naive.tif}"

python -c "
import rasterio
with rasterio.open('${TIF_PATH}') as src:
    print(f'Size: {src.width}x{src.height}')
    print(f'Bands: {src.count}')
    print(f'CRS: {src.crs}')
    print(f'Dtype: {src.dtypes}')
    print(f'Nodata: {src.nodata}')
    print(f'Transform: {src.transform}')
"
