#!/usr/bin/bash

echo "hello world"
python tools/export_onnx.py \
    --cfg_path ./config/nanodet-plus-m-1.5x_416.yml \
    --model_path ./pth/nanodet-plus-m-1.5x_416.pth \
    --out_path ./pth/nanodet.onnx