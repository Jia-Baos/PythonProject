#!/usr/bin/bash

echo "hello world"
python demo/demo.py --demo image \
    --config ./config/nanodet-plus-m-1.5x_416.yml \
    --model ./pth/nanodet-plus-m-1.5x_416.pth \
    --path ./imgs/1.png