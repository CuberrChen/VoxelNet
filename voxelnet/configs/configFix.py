#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 上午9:50
# @Author  : chenxb
# @FileName: configFix.py
# @Software: PyCharm
# @mail ：joyful_chen@163.com

from pathlib import Path
import yaml
from easydict import EasyDict

cfg = EasyDict()


# model
cfg.model = EasyDict()
cfg.model.voxelnet = EasyDict()

cfg.model.voxelnet.voxel_generator = EasyDict()
cfg.model.voxelnet.voxel_generator.point_cloud_range = [0, -40, -3, 70.4, 40, 1]
cfg.model.voxelnet.voxel_generator.voxel_size = [0.2, 0.2, 0.4]
cfg.model.voxelnet.voxel_generator.max_number_of_points_per_voxel = 35

cfg.model.voxelnet.num_class = 1
cfg.model.voxelnet.lidar_input = False # just a flag can ignore it

cfg.model.voxelnet.voxel_feature_extractor = EasyDict()
cfg.model.voxelnet.voxel_feature_extractor.module_class_name = "VoxelFeatureExtractor"
cfg.model.voxelnet.voxel_feature_extractor.num_filters = [32, 128]
cfg.model.voxelnet.voxel_feature_extractor.with_distance = False

cfg.model.voxelnet.middle_feature_extractor = EasyDict()
cfg.model.voxelnet.middle_feature_extractor.module_class_name = "MiddleExtractor"
cfg.model.voxelnet.middle_feature_extractor.num_filters_down1 = [64]
cfg.model.voxelnet.middle_feature_extractor.num_filters_down2 = [64, 64]

cfg.model.voxelnet.rpn = EasyDict()
cfg.model.voxelnet.rpn.module_class_name = "RPN"
cfg.model.voxelnet.rpn.layer_nums = [3, 5, 5]
cfg.model.voxelnet.rpn.layer_strides = [2, 2, 2]
cfg.model.voxelnet.rpn.num_filters = [128, 128, 256]
cfg.model.voxelnet.rpn.upsample_strides = [1, 2, 4]
cfg.model.voxelnet.rpn.num_upsample_filters = [256, 256, 256]
cfg.model.voxelnet.rpn.use_groupnorm = False
cfg.model.voxelnet.rpn.num_groups = 32

cfg.model.voxelnet.loss = EasyDict()
cfg.model.voxelnet.loss.classification_weight = 1.0
cfg.model.voxelnet.loss.localization_weight = 2.0 # 2.0 get better performance

cfg.model.voxelnet.loss.classification_loss = EasyDict()
cfg.model.voxelnet.loss.classification_loss.classification_loss_type = "weighted_sigmoid_focal" #  "weighted_sigmoid" | "weighted_sigmoid_focal"
cfg.model.voxelnet.loss.classification_loss.weighted_sigmoid_focal = EasyDict()
cfg.model.voxelnet.loss.classification_loss.weighted_sigmoid_focal.alpha = 0.25
cfg.model.voxelnet.loss.classification_loss.weighted_sigmoid_focal.gamma = 2.0
cfg.model.voxelnet.loss.classification_loss.weighted_sigmoid_focal.anchorwise_output = True

cfg.model.voxelnet.loss.localization_loss = EasyDict()
cfg.model.voxelnet.loss.localization_loss.localization_loss_type = "weighted_smooth_l1"
cfg.model.voxelnet.loss.localization_loss.weighted_smooth_l1 = EasyDict()
cfg.model.voxelnet.loss.localization_loss.weighted_smooth_l1.sigma =  3.0
cfg.model.voxelnet.loss.localization_loss.weighted_smooth_l1.code_weight =  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# Outputs
cfg.model.voxelnet.use_sigmoid_score = True
cfg.model.voxelnet.encode_background_as_zeros = True
cfg.model.voxelnet.encode_rad_error_by_sin = True

cfg.model.voxelnet.use_direction_classifier = True  # this can help for orientation benchmark.
cfg.model.voxelnet.direction_loss_weight = 0.2  # enough.
cfg.model.voxelnet.use_aux_classifier = False
# Loss
cfg.model.voxelnet.pos_class_weight = 1.5
cfg.model.voxelnet.neg_class_weight = 1.0

cfg.model.voxelnet.loss_norm_type = 1 #1 = NormByNumPositives  0 = LossNormType.NormByNumExamples,  2 = LossNormType.NormByNumPosNeg,


# Postprocess
cfg.model.voxelnet.post_center_limit_range = [0, -40, -5.0, 70.4, 40, 5.0]
cfg.model.voxelnet.use_rotate_nms = True
cfg.model.voxelnet.use_multi_class_nms = False
cfg.model.voxelnet.nms_pre_max_size = 1000
cfg.model.voxelnet.nms_post_max_size = 100
cfg.model.voxelnet.nms_score_threshold = 0.3
cfg.model.voxelnet.nms_iou_threshold = 0.01

cfg.model.voxelnet.use_bev = False
cfg.model.voxelnet.num_point_features = 4
cfg.model.voxelnet.without_reflectivity = False

cfg.model.voxelnet.box_coder = EasyDict()
cfg.model.voxelnet.box_coder.box_coder_type = "ground_box3d_coder"
cfg.model.voxelnet.box_coder.ground_box3d_coder = EasyDict()
cfg.model.voxelnet.box_coder.ground_box3d_coder.linear_dim = False
cfg.model.voxelnet.box_coder.ground_box3d_coder.encode_angle_vector = False

cfg.model.voxelnet.target_assigner = EasyDict()
cfg.model.voxelnet.target_assigner.anchor_generators = EasyDict()
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_type = "anchor_generator_range"
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range = EasyDict()
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.sizes = [1.6, 3.9, 1.56]  # wlh
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.anchor_ranges = [0, -40, -1, 70.4, 40, -1]  # carefully set z center -1
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.rotations = [0, 1.57]  # 0-pi/2
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.matched_threshold = 0.6
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.unmatched_threshold = 0.45
cfg.model.voxelnet.target_assigner.anchor_generators.anchor_generator_range.class_name = "Car"

cfg.model.voxelnet.target_assigner.sample_positive_fraction = -1
cfg.model.voxelnet.target_assigner.sample_size = 512
cfg.model.voxelnet.target_assigner.region_similarity_calculator = EasyDict()
cfg.model.voxelnet.target_assigner.region_similarity_calculator.region_similarity_type = "nearest_iou_similarity"
cfg.model.voxelnet.target_assigner.region_similarity_calculator.nearest_iou_similarity = EasyDict()


# train dataset
cfg.train_input_reader = EasyDict()
cfg.train_input_reader.record_file_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_train.log"
cfg.train_input_reader.class_names = ["Car"]
cfg.train_input_reader.max_num_epochs = 160
cfg.train_input_reader.batch_size = 2  #  use 14.2 GB GPU memory when batch_size=3
cfg.train_input_reader.prefetch_size = 25
cfg.train_input_reader.max_number_of_voxels = 20000  # 6500 to support batchsize=2
cfg.train_input_reader.shuffle_points = True
cfg.train_input_reader.num_workers = 2
cfg.train_input_reader.groundtruth_localization_noise_std = [1.0, 1.0, 1.0] #1.0, 1.0, 1.0 in paper #1.0 1.0 0.5 in second
cfg.train_input_reader.groundtruth_rotation_uniform_noise = [-0.3141592654, 0.3141592654] # -pi/10-pi/10 in paper #-0.78539816, 0.78539816 in second
cfg.train_input_reader.global_rotation_uniform_noise = [-0.78539816, 0.78539816] # -pi/4-pi/4
cfg.train_input_reader.global_scaling_uniform_noise = [0.95, 1.05]
cfg.train_input_reader.global_random_rotation_range_per_object = [0, 0]  # pi/4 ~ 3pi/4
cfg.train_input_reader.anchor_area_threshold = 1
cfg.train_input_reader.remove_points_after_sample = False
cfg.train_input_reader.groundtruth_points_drop_percentage = 0.0
cfg.train_input_reader.groundtruth_drop_max_keep_points = 15
cfg.train_input_reader.use_group_id = False
#the "group_id" is used for data augmentation of objects like tractor/trailer pair.
#we can use group_id to sample/rotate them together. This feature is deprecated, may be removed in future. just ignore it.

cfg.train_input_reader.database_sampler = EasyDict()
cfg.train_input_reader.database_sampler.database_info_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_dbinfos_train.pkl"
cfg.train_input_reader.database_sampler.sample_groups = EasyDict()
cfg.train_input_reader.database_sampler.sample_groups.name_to_max_num = EasyDict()
cfg.train_input_reader.database_sampler.sample_groups.name_to_max_num.key = "Car"
cfg.train_input_reader.database_sampler.sample_groups.name_to_max_num.value = 15

cfg.train_input_reader.database_sampler.database_prep_steps = EasyDict()
cfg.train_input_reader.database_sampler.database_prep_steps.database_preprocessing_step_type = ["filter_by_min_num_points","filter_by_difficulty"] # add prep step by list
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points = EasyDict()
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs = EasyDict()
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs.key = "Car"
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs.value = 5
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_difficulty = EasyDict()
cfg.train_input_reader.database_sampler.database_prep_steps.filter_by_difficulty.removed_difficulties = [-1]

cfg.train_input_reader.database_sampler.global_random_rotation_range_per_object = [0,0]
cfg.train_input_reader.database_sampler.rate = 1.0
cfg.train_input_reader.remove_unknown_examples = False
cfg.train_input_reader.remove_environment = False
cfg.train_input_reader.kitti_info_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_infos_train.pkl"
cfg.train_input_reader.kitti_root_path = "/root/paddlejob/workspace/train_data/datasets/kitti"

# eval dataset
cfg.eval_input_reader = EasyDict()
cfg.eval_input_reader.record_file_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_val.log"
cfg.eval_input_reader.class_names = ["Car"]
cfg.eval_input_reader.batch_size = 2
cfg.eval_input_reader.max_num_epochs  = 160
cfg.eval_input_reader.prefetch_size  = 25
cfg.eval_input_reader.max_number_of_voxels = 20000
cfg.eval_input_reader.shuffle_points = False
cfg.eval_input_reader.num_workers = 2
cfg.eval_input_reader.anchor_area_threshold = 1
cfg.eval_input_reader.remove_unknown_examples = False
cfg.eval_input_reader.remove_environment = False
cfg.eval_input_reader.kitti_info_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_infos_val.pkl"
cfg.eval_input_reader.kitti_root_path = "/root/paddlejob/workspace/train_data/datasets/kitti"
cfg.eval_input_reader.database_sampler = EasyDict()
cfg.eval_input_reader.database_sampler.database_info_path = "/root/paddlejob/workspace/train_data/datasets/kitti/kitti_dbinfos_train.pkl"
cfg.eval_input_reader.database_sampler.sample_groups = EasyDict()
cfg.eval_input_reader.database_sampler.sample_groups.name_to_max_num = EasyDict()
cfg.eval_input_reader.database_sampler.sample_groups.name_to_max_num.key = "Car"
cfg.eval_input_reader.database_sampler.sample_groups.name_to_max_num.value = 15

cfg.eval_input_reader.database_sampler.database_prep_steps = EasyDict()
cfg.eval_input_reader.database_sampler.database_prep_steps.database_preprocessing_step_type = ["filter_by_min_num_points","filter_by_difficulty"] # add prep step by list
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points = EasyDict()
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs = EasyDict()
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs.key = "Car"
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_min_num_points.min_num_point_pairs.value = 5
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_difficulty = EasyDict()
cfg.eval_input_reader.database_sampler.global_random_rotation_range_per_object = [0,0]
cfg.eval_input_reader.database_sampler.database_prep_steps.filter_by_difficulty.removed_difficulties = [-1]
cfg.eval_input_reader.database_sampler.rate = 1.0
cfg.eval_input_reader.groundtruth_localization_noise_std = [1.0, 1.0, 1.0]
cfg.eval_input_reader.groundtruth_rotation_uniform_noise = [-0.3141592654, 0.3141592654]
cfg.eval_input_reader.global_rotation_uniform_noise = [-0.78539816, 0.78539816]
cfg.eval_input_reader.global_scaling_uniform_noise = [0.95, 1.05]
cfg.eval_input_reader.global_random_rotation_range_per_object = [0, 0]  # pi/4 ~ 3pi/4
cfg.eval_input_reader.groundtruth_points_drop_percentage = 0.0
cfg.eval_input_reader.groundtruth_drop_max_keep_points = 15
cfg.eval_input_reader.use_group_id = False
cfg.eval_input_reader.remove_points_after_sample = False

# train config
cfg.train_config = EasyDict()
cfg.train_config.optimizer = EasyDict()
cfg.train_config.optimizer.optimizer_type = "momentum_optimizer" # SGD
cfg.train_config.optimizer.use_moving_average = False

cfg.train_config.optimizer.momentum_optimizer = EasyDict()
cfg.train_config.optimizer.momentum_optimizer.weight_decay = 0.0001
cfg.train_config.optimizer.momentum_optimizer.momentum_optimizer_value = 0.9

cfg.train_config.optimizer.momentum_optimizer.learning_rate = EasyDict()
cfg.train_config.optimizer.momentum_optimizer.learning_rate.learning_rate_type = "polynomial_decay_learning_rate"
cfg.train_config.optimizer.momentum_optimizer.learning_rate.polynomial_decay_learning_rate = EasyDict()
cfg.train_config.optimizer.momentum_optimizer.learning_rate.polynomial_decay_learning_rate.initial_learning_rate = 0.005 # 0.01(bs=16) | 0.002(bs=3)
#0.01*3/16 ~=0.002 #0.0002 in second
cfg.train_config.optimizer.momentum_optimizer.learning_rate.polynomial_decay_learning_rate.decay_steps = 74240 # (bs=8) | 198080(bs=3)
cfg.train_config.optimizer.momentum_optimizer.learning_rate.polynomial_decay_learning_rate.decay_factor = 0.9
cfg.train_config.optimizer.momentum_optimizer.learning_rate.polynomial_decay_learning_rate.end_lr = 0


cfg.train_config.inter_op_parallelism_threads = 4
cfg.train_config.intra_op_parallelism_threads = 4
cfg.train_config.steps = 74240  # 37120: 232 (bs=16) * 160 |74240 (bs=8)*160|198080: 1238(bs=3) *160 | 296960: 1856(bs=2)*160
cfg.train_config.steps_per_eval = 2320  # 232 *5 |464*5|1238 * 5 |1856 *5
# steps = 296960 # 1857 * 160
# steps_per_eval = 9280 # 1856 * 5
cfg.train_config.save_checkpoints_secs = 3600  # one hour
cfg.train_config.save_summary_steps = 10
cfg.train_config.enable_mixed_precision = False # dont support now
cfg.train_config.loss_scale_factor = 512.0 #dont support now
cfg.train_config.clear_metrics_every_epoch = False