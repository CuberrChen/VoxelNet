#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午9:33
# @Author  : chenxb
# @FileName: sample_infer.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import paddle
import voxelnet.pypaddle.builder.voxelnet_builder as voxelnet_builder
import voxelnet.builder.voxel_builder as voxel_builder
import voxelnet.builder.target_assigner_builder as target_assigner_builder
import voxelnet.pypaddle.builder.box_coder_builder as box_coder_builder
from voxelnet.configs import cfg_from_config_py_file
from voxelnet.utils import vis

paddle.set_device('gpu') # 设置cpu/gpu

config_path = "home/aistudio/VoxelNet/voxelnet/configs/config.py"
config = cfg_from_config_py_file(config_path)
input_cfg = config.eval_input_reader
model_cfg = config.model.voxelnet

ckpt_path = "/home/aistudio/output_exp1229/voxelnet-198080.ckpt"
voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
######################
# BUILD TARGET ASSIGNER
######################
bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
box_coder = box_coder_builder.build(model_cfg.box_coder)
target_assigner_cfg = model_cfg.target_assigner
target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                bv_range, box_coder)
net = voxelnet_builder.build(model_cfg, voxel_generator, target_assigner)
net.eval()

state = paddle.load(ckpt_path)
net.set_state_dict(state)

out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
grid_size = voxel_generator.grid_size
feature_map_size = grid_size[:2] // out_size_factor
feature_map_size = [*feature_map_size, 1][::-1]

anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
anchors = paddle.to_tensor(anchors, dtype=paddle.float32)
anchors = anchors.reshape((1, -1, 7))

info_path = input_cfg.kitti_info_path
root_path = Path(input_cfg.kitti_root_path)
with open(info_path, 'rb') as f:
    infos = pickle.load(f)

info = infos[1] # 测试目标点云
v_path = info['velodyne_path']
v_path = str(root_path / v_path)
points = np.fromfile(
    v_path, dtype=np.float32, count=-1).reshape([-1, 4])
voxels, coords, num_points = voxel_generator.generate(points, max_voxels=20000)
print(voxels.shape)
# add batch idx to coords
coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
voxels = paddle.to_tensor(voxels, dtype=paddle.float32)
coords = paddle.to_tensor(coords, dtype=paddle.int32)
num_points = paddle.to_tensor(num_points, dtype=paddle.int32)

example = {
    "anchors": anchors,
    "voxels": voxels,
    "num_points": num_points,
    "coordinates": coords,
}
with paddle.no_grad():
    pred = net(example)

boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
vis_voxel_size = [0.1, 0.1, 0.1]
vis_point_range = [-50, -30, -3, 50, 30, 1]
bev_map = vis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
bev_map = vis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)

plt.imshow(bev_map)