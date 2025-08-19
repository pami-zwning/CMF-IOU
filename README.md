# CMF-IOU

It is the official code release of [CMF-IOU (TCSVT 2025)](https://arxiv.org/abs/2508.12917v1). We design a multistage cross-modal fusion 3D detection framework, termed CMF-IOU, to effectively address the challenge of aligning 3D spatial and 2D semantic information. The code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Framework
![](./tools/images/framework.png)
Overview of our CMF-IOU framework. We estimate the pseudo points by a depth completion network and integrate them with the LiDAR points as
our input. (a) The bilateral cross-view enhancement backbone contains the S2D branch and the ResVC branch, where the S2D branch encodes the raw voxel
features and the ResVC branch encodes the pseudo voxel features. (b) The iterative voxel-point aware fine grained pooling is designed for optimizing the
predicted and generated proposals. (c) The IoU joint prediction balances the IoU and classification scores in the NMS post-processing stage.


## Getting Started
```
conda create -n cmfiou python=3.9
conda activate cmfiou
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
python setup.py develop
```

## Dataset Preparation
We generate the pseudo points via the depth completion model [PENet](https://arxiv.org/abs/2103.00783) and [MVP](https://arxiv.org/abs/2111.06881).

For the KITTI dataset, we employ the [PENet](https://arxiv.org/abs/2103.00783) model to generate pseudo points, using the following dataset configuration:

* Generate the raw and pseudo data infos by running the following command:
```
python pcdet.datasets.kitti_dataset_mm create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
```
kitti
 |---ImageSets
      |---train.txt
      |---val.txt
      |---test.txt
 |---gt_database (optional)
 |---gt_database_mm
 |---training
      |---calib & label_2 & image_2 & velodyne & velodyne_depth & planes (optional)
 |---testing
      |---calib & image_2 & velodyne & velodyne_depth
 |---kitti_dbinfos_train_mm.pkl
 |---kitti_infos_train.pkl
 |---kitti_infos_val.pkl
 |---kitti_infos_test.pkl
```

## Running Command
```
cd tools & bash run_cmfiou_kitti.sh
```

### Results on KITTI validation dataset (mAP)
| Category | 3D_Easy | 3D_Mod | 3D_Hard | BEV_Easy | BEV_Mod | BEV_Hard | Config |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Car | 91.92 | 85.14 | 80.63 | 95.72 | 92.07 | 87.25 | [CMF-IOU-KITTI.yaml](./tools/cfgs/models/kitti/CMF-IOU-MM.yaml) |
| Pedestrain | 53.26 | 49.23 | 46.19 | 58.62 | 52.48 | 50.31 | [CMF-IOU-KITTI.yaml](./tools/cfgs/models/kitti/CMF-IOU-MM.yaml) |
| Cyclist | 85.56 | 71.84 | 63.43 | 87.21 | 72.87 | 66.85 | [CMF-IOU-KITTI.yaml](./tools/cfgs/models/kitti/CMF-IOU-MM.yaml) |


### Results on nuScenes dataset
| set | mAP | NDS |  Car  | Truck | C.V. | Bus | Trailer | Barrier | Motor. | Bicycle | Ped. | T.C. |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| validation | 69.1 | 72.4 | 88.2 | 62.4 | 28.9 | 77.5 | 48.5 | 71.3 | 77.2 | 64.7 | 90.1 | 82.1 |
| testing | 69.8 | 72.6 | 87.5 | 59.0 | 36.2 | 69.5 | 65.1 | 78.9 | 74.9 | 48.2 | 90.4 | 88.3 |

<!-- ## TODO
* [ ] Add the details of the pseudo point clouds generation.
* [ ] Release the weights or checkpoints of our model. -->


## Citation
```
@misc{ning2025cmfioumultistagecrossmodalfusion,
      title={CMF-IoU: Multi-Stage Cross-Modal Fusion 3D Object Detection with IoU Joint Prediction}, 
      author={Zhiwei Ning and Zhaojiang Liu and Xuanang Gao and Yifan Zuo and Jie Yang and Yuming Fang and Wei Liu},
      year={2025},
      eprint={2508.12917},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.12917}, 
}
```

## Acknowleadgement
This work is partially supported by National Natural Science Foundation of China (Grant No. 62376153, 62402318, 24Z990200676, 62271237, U24A20220，62132006，62311530101) and Science Foundation of the Jiangxi Province of China (Grant No. 20242BAB26014).
