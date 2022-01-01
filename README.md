# Voxelnet
Voxelnet detector. Based on my PaddlePaddle implementation of VoxelNet.

### Performance in KITTI validation set (50/50 split, people have problems, need to be tuned.)

```
Car AP@0.70, 0.70, 0.70:
bbox AP:88.58, 78.71, 77.12
bev  AP:88.57, 84.54, 77.97
3d   AP:72.27, 62.04, 55.51
aos  AP:40.35, 35.18, 33.44
Car AP@0.70, 0.50, 0.50:
bbox AP:88.58, 78.71, 77.12
bev  AP:90.24, 88.79, 87.48
3d   AP:90.03, 87.99, 86.29
aos  AP:40.35, 35.18, 33.44

Car coco AP@0.50:0.05:0.95:
bbox AP:63.13, 58.65, 56.50
bev  AP:65.08, 61.62, 58.64
3d   AP:49.81, 45.11, 42.36
aos  AP:28.89, 26.27, 24.59
```

## Install

### 1. Clone code

```bash
git clone https://github.com/CuberrChen/VoxelNet.git
cd ./VoxelNet/voxelnet
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

```bash
pip install shapely fire pybind11 protobuf scikit-image numba pillow
```

If you don't have Anaconda:

```bash
pip install numba
```


### 3. Setup cuda for numba

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. add VoxelNet/ to PYTHONPATH

## Prepare dataset

* Dataset preparation

Download KITTI dataset and create some directories first:

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

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```

## Usage

### train

```bash
python ./pypaddle/train.py train --config_path=./configs/config.py --model_dir=/path/to/model_dir
```
```
python -m paddle.distributed.launch ./pypaddle/train_mgpu.py --config_path=./configs/config.py --model_dir=/path/to/model_dir

```
### evaluate

```bash
python ./pypaddle/train.py evaluate --config_path=./configs/config.py --model_dir=/path/to/model_dir
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### pretrained model's sample inference

details in ./pypaddle/sample_infer.py
```
python ./pypaddle/sample_infer.py --config_path=./configs/config.py --checkpoint_path=/path/to/../**.ckpt --index 564
```
you can test pointcloud and visualize its BEV result.

retust picture:

![bev result](images/val564.png)
## Try Kitti Viewer Web

### Major step

1. run ```python ./kittiviewer/backend.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/CuberrChen/VoxelNet/main/images/viewerweb.png)



## Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](https://raw.githubusercontent.com/CuberrChen/VoxelNet/main/images/simpleguide.png)

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/CuberrChen/VoxelNet/main/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].