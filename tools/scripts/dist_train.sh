#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=20017 train.py --launcher pytorch ${PY_ARGS} > log_v2.txt&
