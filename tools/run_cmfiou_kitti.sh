cfg_name=CMF-IOU-MM

## kitti mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/models/kitti/$cfg_name.yaml \
--extra_tag CMF-IOU-MM \
--batch_size 4  --epochs 20 --max_ckpt_save_num 5 --workers 8 --sync_bn