[README中文](README.md)|README_EN
# VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

## 1 Introduction
![images](images/network.png)  
This project replicates VoxelNet, a voxel-based 3D target detection algorithm based on the PaddlePaddle framework, and experiments on the KITTI data set.
The project provides pre-trained models and AiStudio online experience with NoteBook.

**Paper：**
- [1] Yin Zhou, Oncel Tuzel.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396)

**Project Reference：**
- [https://github.com/qianguih/voxelnet](https://github.com/qianguih/voxelnet) 

    The repo performance: easy: 53.43 moderate:48.78 hard:48.06
    
- [https://github.com/traveller59/second.pytorch](https://github.com/traveller59/second.pytorch)

Since the paper does not provide open source code, no project can be found to reproduce the metrics in the paper.
Therefore, this project is based on the reference project (voxelnet-tensorflow) and the subsequent improved version of the algorithm (second) of the paper.


## 2 Performance
>The results on the KITTI val dataset (50/50 split as paper) are shown in the table below。

1、When the network structure and loss function as well as most of the data processing and training configuration are the same as the original paper, the weight distribution (1:1 in the paper, 1:2 here is better after experiment) and batch size and learning rate of cls loss and loc loss are different.
The achieved results are shown in the following table：

|NetWork |epochs|opt|lr|batch_size|dataset|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|VoxelNet|160|SGD|0.00125|2 * 1(card)|KITTI|[config](voxelnet/configs/config.py)|

```
Car AP@0.70, 0.70, 0.70:
bbox AP:88.95, 78.93, 77.26
bev  AP:89.06, 79.08, 77.82
3d   AP:78.21, 63.72, 56.60
aos  AP:43.70, 39.11, 37.70
Car AP@0.70, 0.50, 0.50:
bbox AP:88.95, 78.93, 77.26
bev  AP:90.52, 88.58, 87.35
3d   AP:90.35, 87.57, 86.29
aos  AP:43.70, 39.11, 37.70

Car coco AP@0.50:0.05:0.95:
bbox AP:65.77, 59.51, 57.18
bev  AP:65.98, 61.36, 58.41
3d   AP:52.82, 46.07, 43.27
aos  AP:33.11, 29.14, 27.63
```
Pre-trained weights and training log：[]()

2、The results are significantly improved when the CrossEntropy loss is changed to FocalLoss and when the direction classification loss for aos is added

|NetWork |epochs|opt|lr|batch_size|dataset|config|
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
|VoxelNet|160|SGD|0.005|2 * 4(card)|KITTI|[configFix](voxelnet/configs/configFix.py)|
```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.21, 85.07, 79.22
bev  AP:89.83, 84.61, 78.87
3d   AP:80.28, 66.42, 62.63
aos  AP:89.72, 83.71, 77.44
Car AP@0.70, 0.50, 0.50:
bbox AP:90.21, 85.07, 79.22
bev  AP:95.96, 89.35, 88.39
3d   AP:90.66, 88.90, 87.31
aos  AP:89.72, 83.71, 77.44

Car coco AP@0.50:0.05:0.95:
bbox AP:66.58, 62.54, 60.02
bev  AP:68.17, 62.85, 60.12
3d   AP:54.16, 48.89, 46.22
aos  AP:66.24, 61.58, 58.61
```
Pre-trained weights and training log：[]()

**In addition, the details not mentioned in the paper, this project are referred to the implementation of the Second project**

## 3 Start

### 1. clone

```bash
git clone https://github.com/CuberrChen/VoxelNet.git
```

### 2. Dependencies
The most suitable environment configuration：
- **python version**：3.7.4
- **PaddlePaddle version**：2.2.1
- **CUDA version**： NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: **11.0**   cuDNN:7.6

Note：
**Due to PaddlePaddle/cuDNN's BUG, when CUDA 10.1, batch size > 2, there is a error**：
```
OSError: (External) CUDNN error(7), CUDNN_STATUS_MAPPING_ERROR. 
  [Hint: 'CUDNN_STATUS_MAPPING_ERROR'.  An access to GPU memory space failed, which is usually caused by a failure to bind a texture.  To correct, prior to the function call, unbind any previously bound textures.  Otherwise, this may indicate an internal error/bug in the library.  ] (at /paddle/paddle/fluid/operators/conv_cudnn_op.cu:758)

```

Therefore, if the single card environment is not CUDA 11.0 or above, the batch size in the config file can be set to 2. Subsequently, the gradient accrual is turned on by the accum_step parameter of training to increase the effect of bs. Set accum_step=8 that means bs=16, and do the initial learning rate adjustment of the corresponding config file.
- Dependency Package Installation：
```bash
cd VoxelNet/
pip install -r requirements.txt
```

### 3. Setting up the cuda environment for numba

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. add VoxelNet/ to PYTHONPATH
```bash
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/VoxelNet
```
## 4 Datasets

* Dataset preparation

Fristly, Download [KITTI 3D Object Det](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and create some folders:

```plain
└── KITTI_DATASET_ROOT 
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

* Create kitti infos:

```bash
cd ./VoxelNet/voxelnet
```

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

* Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

* Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

* configs/config.py to fix config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl" # 比如 /home/aistudio/data/kitti/kitti_dbinfos_train.pkl
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl" 
  kitti_root_path: "KITTI_DATASET_ROOT"
# 比如 kitti_info_path: "/home/aistudio/data/kitti/kitti_infos_train.pkl"
# 比如 kitti_root_path: "/home/aistudio/data/kitti"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```

Setting Notes.
If the gradient accumulation option is to be turned on for training.
- The decay_steps of the learning rate is set according to the total steps corresponding to the batch size after **gradient accumulation**.
- train_config.steps is set according to the total steps corresponding to the initial batch size when **no gradient accrual is applied**.

## 5 Quick Start

### Train

```bash
python ./pypaddle/train.py train --config_path=./configs/config.py --model_dir=./output
```

```
python -m paddle.distributed.launch ./pypaddle/train_mgpu.py --config_path=./configs/config.py --model_dir=./output
```
### Evaluate

```bash
python ./pypaddle/train.py evaluate --config_path=./configs/config.py --model_dir=./output
```

* The detection results are saved as a result.pkl file to model_dir/eval_results/step_xxx or to the official KITTI label format if you specify --pickle_result=False.
* You can specify the pre-trained model you want to evaluate with --ckpt_path=path/***.ckpt, if not specified, the latest model will be found in the model_dir folder containing the training generated json files by default.
### Pretrained Model's Sample Inference

details in ./pypaddle/sample_infer.py
```
python ./pypaddle/sample_infer.py --config_path=./configs/config.py --checkpoint_path=./output/**.ckpt --index 564
```
you can test pointcloud and visualize its BEV result.

retust picture:

![bev result](images/val564.png)

Note：Points that are projectedoutside of image boundaries are removed(in Paper Section 3.1)),So there is no detection frame behind the body.

## 3D Visualization

### 1. Try Kitti Viewer Web

#### Major step

1. run ```python ./kittiviewer/backend.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

#### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](images/viewerweb.png)



### 2. Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](images/simpleguide.png)

## Concepts

* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].


## Model Information

Related Information:

| Information | Description |
| --- | --- |
| Author | xbchen|
| Date | 2021.1 |
| Framework | PaddlePaddle>=2.2.1 |
| Scenarios | 3D target detection |
| Hardware | GPU |
| Online | [Notebook](https://aistudio.baidu.com/aistudio/projectdetail/3291060?contributionType=1)|
| Multi-card | [Shell](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3369575)|

## Citation
- Thanks for yan yan's [second.pytorch project](https://github.com/traveller59/second.pytorch).

```
@inproceedings{Yin2018voxelnet,
    author={Yin Zhou, Oncel Tuzel},
    title={VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2018}
}
```