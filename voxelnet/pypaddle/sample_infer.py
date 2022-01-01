#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午9:33
# @Author  : chenxb
# @FileName: sample_infer.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

import paddle
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
import voxelnet.pypaddle.builder.voxelnet_builder as voxelnet_builder
import voxelnet.builder.voxel_builder as voxel_builder
import voxelnet.builder.target_assigner_builder as target_assigner_builder
import voxelnet.pypaddle.builder.box_coder_builder as box_coder_builder
from voxelnet.data.preprocess import merge_voxelnet_batch
from voxelnet.configs import cfg_from_config_py_file
from voxelnet.utils import vis


def example_convert_to_paddle(example, dtype=paddle.float32,
                             ) -> dict:
    example_paddle = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_paddle[k] = paddle.to_tensor(v, dtype=dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.int32)
        elif k in ["anchors_mask"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.uint8)
        else:
            example_paddle[k] = v
    return example_paddle

def main(args):
    paddle.set_device(args.device) # 设置cpu/gpu

    config_path = args.config_path
    config = cfg_from_config_py_file(config_path)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.voxelnet

    ckpt_path = args.checkpoint_path
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
    anchors = anchors.reshape((1, -1, 7))

    info_path = input_cfg.kitti_info_path
    root_path = Path(input_cfg.kitti_root_path)
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    info = infos[args.index] # 测试目标index in kitti [val.txt]
    v_path = info['velodyne_path']
    v_path = str(root_path / v_path)
    points = np.fromfile(
        v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    voxels, coords, num_points = voxel_generator.generate(points, max_voxels=20000)
    print(voxels.shape)
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
        "coordinates": coords,
        "rect": rect,
        "P2": P2,
        "Trv2c":Trv2c,
        'image_idx': image_idx

    }
    batch_example = [example]
    examples = merge_voxelnet_batch(batch_example)
    examples = example_convert_to_paddle(examples)
    with paddle.no_grad():
        pred = net(examples)[0]

    boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = vis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = vis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)

    plt.imshow(bev_map)


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')
    # params of training
    parser.add_argument(
        '--config_path', dest="config_path", help="The config file.", default="./configs/config.py", type=str)
    parser.add_argument(
        '--checkpoint_path',
        dest='checkpoint_path',
        help='The directory for saving the model snapshot',
        type=str,
        default="./voxelnet-198080.ckpt")
    parser.add_argument(
        '--index',
        dest='index',
        help='The index of test pointcloud in list([val])',
        type=int,
        default=564)
    parser.add_argument(
        '--device',
        dest='device',
        help='The device',
        type=str,
        default='gpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)