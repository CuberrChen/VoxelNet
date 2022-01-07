README中文|[README_EN](README_EN.md)
# VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

## 1 简介
![images](images/network.png)  
本项目基于PaddlePaddle框架复现了基于体素的3D目标检测算法VoxelNet，在KITTI据集上进行了实验。
项目提供预训练模型和AiStudio在线体验NoteBook。

**论文：**
- [1] Yin Zhou, Oncel Tuzel.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396)

**项目参考：**
- [https://github.com/qianguih/voxelnet](https://github.com/qianguih/voxelnet) 

    github repo实现精度 easy: 53.43 moderate:48.78 hard:48.06
    
- [https://github.com/traveller59/second.pytorch](https://github.com/traveller59/second.pytorch)

由于该论文并未提供开源的代码，目前也找不到能够复现其论文中指标的项目。
因此本项目根据参考项目（voxelnet-tensorflow）和该论文后续的算法改进版本（second）进行了复现。


## 2 复现精度
>在KITTI val数据集（50/50 split as paper）的测试效果如下表。

1、当网络结构和损失函数以及大部分数据处理、训练配置和论文一致时，batch size以及学习率不同。
所能达到的结果如下表所示：

|NetWork |epochs|opt|lr|batch_size|dataset|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|VoxelNet|160|SGD|0.0015|2 * 1(V100 card)|KITTI|[config](voxelnet/configs/config.py)|

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
预训练权重和日志：[百度网盘](https://pan.baidu.com/s/1MQ9do53CJHEjtXoD1eSTCg?pwd=fmxk) | [AiStudio存储](https://aistudio.baidu.com/aistudio/datasetdetail/124683)

2、当将分类损失改为FocalLoss以及加入针对aos的direction分类损失时(后续实验表明direction损失只对aos起作用，可不用)

|NetWork |epochs|opt|lr|batch_size|dataset|config|
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
|VoxelNet|160|SGD|0.005|2 * 4 (V100 card)|KITTI|[configFix](voxelnet/configs/configFix.py)|
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
预训练权重和训练日志：[百度网盘](https://pan.baidu.com/s/1LuB5N_CbzWT5HyFDm-a66g?pwd=3633) | [AiStudio存储](https://aistudio.baidu.com/aistudio/datasetdetail/124650)

* **另外，论文中没提及的细节，本项目均参考Second项目的实施**。

* 仓库内的log文件夹下存放有两个训练日志和可视化曲线日志。


## 3 开始

### 1. 克隆项目

```bash
git clone git@github.com:CuberrChen/VoxelNet.git
```

项目结构：

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

### 2. 安装依赖
最适合的环境配置：
- **python版本**：3.7.4
- **PaddlePaddle框架版本**：2.2.1
- **CUDA 版本**： NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: **11.0**   cuDNN:7.6

注意：
**由于PaddlePaddle/cuDNN本身的BUG，CUDA 10.1版本当batch size > 2时会报如下错误**：
```
OSError: (External) CUDNN error(7), CUDNN_STATUS_MAPPING_ERROR. 
  [Hint: 'CUDNN_STATUS_MAPPING_ERROR'.  An access to GPU memory space failed, which is usually caused by a failure to bind a texture.  To correct, prior to the function call, unbind any previously bound textures.  Otherwise, this may indicate an internal error/bug in the library.  ] (at /paddle/paddle/fluid/operators/conv_cudnn_op.cu:758)

```

因此单卡**如果环境不是CUDA 11.0以上，config文件中batch size设置为2即可，后续通过训练的accum_step参数开启梯度累加起到增大bs的效果**。设置accum_step=8即表示bs=16，并做相应config文件的初始学习率调整。

- 依赖包安装：
```bash
cd VoxelNet/
pip install -r requirements.txt
```

### 3. 为numba设置cuda环境

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. 将项目VoxelNet/添加到Python环境
```bash
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/VoxelNet
```
## 4 数据集

* Dataset preparation

首先下载 [官方KITTI 3D目标检测的数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 或者AiStudio上的数据集：[kitti_detection](https://aistudio.baidu.com/aistudio/datasetdetail/50186) 并创建一些文件夹:

```plain
└── KITTI_DATASET_ROOT # KITTI数据集的路径
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

* 在configs/config.py修改config file

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

设置注意事项：

1、若训练要开启梯度累加选项，则：
- 学习率的decay_steps按照**梯度累加后**的batch size对应的总steps来设置。
- train_config.steps则按**未梯度累加时**对应的初始batch size对应的总steps来设置

2、 配置文件需放置于voxelnet/configs/***.py

## 5 快速开始

### Train

单卡：
```bash
python ./pypaddle/train.py train --config_path=./configs/config.py --model_dir=./output
```
多卡：
```
python -m paddle.distributed.launch ./pypaddle/train_mgpu.py --config_path=./configs/configFix.py --model_dir=./output
```
注意：

* batch size 2 时，单卡训练显存大约11G。可通过修改post_center_limit_range Z的范围以及max_number_of_voxels大小节省显存。

### Evaluate
```bash
python ./pypaddle/train.py evaluate --config_path=./configs/config.py --model_dir=./output
```
* 检测结果会保存成一个 result.pkl 文件到 model_dir/eval_results/step_xxx 或者 保存为官方的KITTI label格式如果指定--pickle_result=False.
* 你可以使用--ckpt_path=path/***.ckpt 指定你想评估的预训练模型，如果不指定，默认在包含训练产生json文件的model_dir文件夹中找最新的模型。

例如：使用上述提供的预训练模型[百度网盘](https://pan.baidu.com/s/1LuB5N_CbzWT5HyFDm-a66g?pwd=3633) | [AiStudio存储](https://aistudio.baidu.com/aistudio/datasetdetail/124650) 进行评估
将下载好的模型参数放置于voxelnet/output文件夹下,配置文件pipeline.py放置于voxelnet/configs文件夹下。
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

注意：由于只保留了相机视角范围内的结果(Points that are projectedoutside of image boundaries are removed(in Paper Section 3.1))，所以车身后面没有检测框。

## 3D可视化

提供了两个用于查看3D点云空间下的可视化工具，一个是Web界面的可视化交互界面，一个是本地端的Viewer.

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


## 模型信息

相关信息:

| 信息 | 描述 |
| --- | --- |
| 作者 | xbchen|
| 日期 | 2021年1月 |
| 框架版本 | PaddlePaddle>=2.2.1 |
| 应用场景 | 3D目标检测 |
| 硬件支持 | GPU |
| 在线体验 | [Notebook](https://aistudio.baidu.com/aistudio/projectdetail/3291060?contributionType=1)|
| 多卡脚本 | [Shell](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3369575)|

## 引用
- Thanks for yan yan's [second.pytorch project](https://github.com/traveller59/second.pytorch).

```
@inproceedings{Yin2018voxelnet,
    author={Yin Zhou, Oncel Tuzel},
    title={VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2018}
}
```