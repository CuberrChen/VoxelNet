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
|VoxelNet|160|SGD|0.0015|2 * 1(card)|KITTI|[config](voxelnet/configs/config.py)|

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.26, 86.24, 79.26
bev  AP:89.92, 86.04, 79.14
3d   AP:77.00, 66.40, 63.24
aos  AP:38.34, 37.30, 33.19
Car AP@0.70, 0.50, 0.50:
bbox AP:90.26, 86.24, 79.26
bev  AP:90.80, 89.84, 88.88
3d   AP:90.75, 89.32, 87.84
aos  AP:38.34, 37.30, 33.19

Car coco AP@0.50:0.05:0.95:
bbox AP:67.72, 63.70, 61.10
bev  AP:67.13, 63.44, 61.15
3d   AP:53.45, 48.92, 46.34
aos  AP:28.82, 27.54, 25.55
```
Pre-trained weights and training log：[Baidu Cloud](https://pan.baidu.com/s/1MQ9do53CJHEjtXoD1eSTCg?pwd=fmxk) | [AiStudio](https://aistudio.baidu.com/aistudio/datasetdetail/124683)

2、The results are improved when the CrossEntropy loss is changed to FocalLoss and when the direction classification loss for aos is added

|NetWork |epochs|opt|lr|batch_size|dataset|config|
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
|VoxelNet|160|SGD|0.005|2 * 4(card)|KITTI|[configFix](voxelnet/configs/configFix.py)|
```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.19, 85.78, 79.38
bev  AP:89.79, 85.26, 78.93
3d   AP:81.78, 66.88, 63.51
aos  AP:89.81, 84.55, 77.71
Car AP@0.70, 0.50, 0.50:
bbox AP:90.19, 85.78, 79.38
bev  AP:96.51, 89.53, 88.59 
3d   AP:90.65, 89.08, 87.52
aos  AP:89.81, 84.55, 77.71

Car coco AP@0.50:0.05:0.95:
bbox AP:67.15, 63.05, 60.58
bev  AP:68.90, 63.78, 61.08
3d   AP:54.88, 49.42, 46.82
aos  AP:66.89, 62.19, 59.23
```
Pre-trained weights and training log：[Baidu Cloud](https://pan.baidu.com/s/1LuB5N_CbzWT5HyFDm-a66g?pwd=3633) | [AiStudio](https://aistudio.baidu.com/aistudio/datasetdetail/124650)

**In addition, this project are referred to the implementation of the Second project for the details not mentioned in the paper, **

## 3 Start

### 1. clone

```bash
git clone https://github.com/CuberrChen/VoxelNet.git
```
project structure:
```
VoxelNet/
├── images
├── log
├── paddleplus
│   ├── nn
│   ├── ops
│   ├── train
│   ├── __init__.py
│   ├── metrics.py
│   └── tools.py
├── README_EN.md
├── README.md
├── requirements.txt
└── voxelnet
    ├── builder
    ├── configs
    ├── core
    ├── data
    ├── kittiviewer
    ├── output
    ├── pypaddle
    ├── utils
    ├── __init__.py
    └── create_data.py
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

Fristly, Download Official [KITTI 3D Object Det](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) or AiStudio [kitti_detection](https://aistudio.baidu.com/aistudio/datasetdetail/50186) and create some folders:

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

1\If the gradient accumulation option is to be turned on for training.
- The decay_steps of the learning rate is set according to the total steps corresponding to the batch size after **gradient accumulation**.
- train_config.steps is set according to the total steps corresponding to the initial batch size when **no gradient accrual is applied**.

2\The configuration file should be placed in voxelnet/configs/***.py

## 5 Quick Start

### Train

```bash
python ./pypaddle/train.py train --config_path=./configs/config.py --model_dir=./output
```

```
python -m paddle.distributed.launch ./pypaddle/train_mgpu.py --config_path=./configs/config.py --model_dir=./output
```
Note:

* The training memory is about 11G for batch size 2. You can save memory by modifying the range of post_center_limit_range Z and the size of max_number_of_voxels.

### Evaluate

```bash
python ./pypaddle/train.py evaluate --config_path=./configs/config.py --model_dir=./output
```

* The detection results are saved as a result.pkl file to model_dir/eval_results/step_xxx or to the official KITTI label format if you specify --pickle_result=False.
* You can specify the pre-trained model you want to evaluate with --ckpt_path=path/***.ckpt, if not specified, the latest model will be found in the model_dir folder containing the training generated json files by default.

For example: Using the pre-trained model provided above [Baidu Cloud](https://pan.baidu.com/s/1LuB5N_CbzWT5HyFDm-a66g?pwd=3633) | [AiStudio](https://aistudio.baidu.com/aistudio/datasetdetail/124650) .

Place the downloaded model parameters in voxelnet/output. Place pipeline.py in voxelnet/configs
```bash
python ./pypaddle/train.py evaluate --config_path=./configs/pipeline.py --model_dir=./output --ckpt_path=./output/voxelnet-73601.ckpt
```

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