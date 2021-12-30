import time
from enum import Enum
from functools import reduce

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

import paddleplus
from paddleplus import metrics
from paddleplus.nn import Empty, GroupNorm
from paddleplus.ops.array_ops import gather_nd, scatter_nd
from paddleplus.tools import change_default_args
from voxelnet.pypaddle.core import box_paddle_ops
from voxelnet.pypaddle.core.losses import (WeightedSigmoidClassificationLoss,
                                           WeightedSmoothL1LocalizationLoss,
                                           WeightedSoftmaxClassificationLoss)


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).astype(cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_neg_loss = (labels == 0).astype(cls_loss.dtype) * cls_loss.reshape((
            batch_size, -1))
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = paddle.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = paddle.arange(
        max_num, dtype=paddle.int32).reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.astype(paddle.int32) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1D = change_default_args(
                epsilon=1e-3, momentum=0.01)(nn.BatchNorm1D)
            Linear = change_default_args(bias_attr=False)(nn.Linear)
        else:
            BatchNorm1D = Empty
            Linear = change_default_args(bias_attr=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1D(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2,1))
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = paddle.max(pointwise, axis=1, keepdim=True)
        # [K, 1, units]
        repeated = aggregated.tile((1, voxel_count, 1))

        concatenated = paddle.concat([pointwise, repeated], axis=2)
        # [K, T, 2 * units]
        return concatenated


class VoxelFeatureExtractor(nn.Layer):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1D = change_default_args(
                epsilon=1e-3, momentum=0.01)(nn.BatchNorm1D)
            Linear = change_default_args(bias_attr=False)(nn.Linear)
        else:
            BatchNorm1D = Empty
            Linear = change_default_args(bias_attr=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features ->7
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        self.norm = BatchNorm1D(num_filters[1])

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            axis=1, keepdim=True) / num_voxels.astype(features.dtype).reshape((-1, 1, 1))
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = paddle.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = paddle.concat(
                [features, features_relative, points_dist], axis=-1)
        else:
            features = paddle.concat([features, features_relative], axis=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.unsqueeze(mask, -1).astype(features.dtype)
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.transpose((0, 2, 1))).transpose((0, 2,1))
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = paddle.max(x, axis=1)
        return voxelwise


class VoxelFeatureExtractorV2(nn.Layer):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1D = change_default_args(
                epsilon=1e-3, momentum=0.01)(nn.BatchNorm1D)
            Linear = change_default_args(bias_attr=False)(nn.Linear)
        else:
            BatchNorm1D = Empty
            Linear = change_default_args(bias_attr=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.LayerList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        self.norm = BatchNorm1D(num_filters[-1])

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            axis=1, keepdim=True) / num_voxels.astype(features.dtype).reshape((-1, 1, 1))
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = paddle.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = paddle.concat(
                [features, features_relative, points_dist], axis=-1)
        else:
            features = paddle.concat([features, features_relative], axis=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.unsqueeze(mask, -1).astype(features.dtype)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.transpose((0, 2, 1))).transpose((
            0, 2, 1))
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = paddle.max(features, axis=1)
        return voxelwise

class ZeroPad3D(nn.Pad3D):
    def __init__(self, padding):
        super(ZeroPad3D, self).__init__(padding, mode='constant',value=0.0)

class ZeroPad2D(nn.Pad2D):
    def __init__(self, padding):
        super(ZeroPad2D, self).__init__(padding, mode='constant',value=0.0)

class MiddleExtractor(nn.Layer):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm3D = change_default_args(
                epsilon=1e-3, momentum=0.01)(nn.BatchNorm3D)
            Conv3D = change_default_args(bias_attr=False)(nn.Conv3D)
        else:
            BatchNorm3D = Empty
            Conv3D = change_default_args(bias_attr=True)(nn.Conv3D)
        self.voxel_output_shape = output_shape
        self.middle_conv = nn.Sequential(
            Conv3D(num_input_features, 64, 3, stride=(2, 1, 1), padding=(1,1,1)),
            BatchNorm3D(64),
            nn.ReLU(),
            Conv3D(64, 64, 3, stride=(1,1,1), padding=(0,1,1)),
            BatchNorm3D(64),
            nn.ReLU(),
            Conv3D(64, 64, 3, stride=(2, 1, 1),padding=(1,1,1)),
            BatchNorm3D(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = paddle.scatter_nd(coors.astype(paddle.int64), voxel_features, output_shape) # TODO
        # print('scatter_nd fw:', time.time() - t)
        ret = ret.transpose((0, 4, 1, 2, 3))
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.reshape((N, C * D, H, W))

        return ret


class RPN(nn.Layer):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2D = change_default_args(
                    num_groups=num_groups, epsilon=1e-3)(GroupNorm)
            else:
                BatchNorm2D = change_default_args(
                    epsilon=1e-3, momentum=0.01)(nn.BatchNorm2D)
            Conv2D = change_default_args(bias_attr=False)(nn.Conv2D)
            Conv2DTranspose = change_default_args(bias_attr=False)(
                nn.Conv2DTranspose)
        else:
            BatchNorm2D = Empty
            Conv2D = change_default_args(bias_attr=True)(nn.Conv2D)
            Conv2DTranspose = change_default_args(bias_attr=True)(
                nn.Conv2DTranspose)

        # note that when stride > 1, Conv2D with same padding isn't
        # equal to pad-Conv2D. we should use pad-Conv2D.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = nn.Sequential(
                Conv2D(6, 32, 3, padding=1),
                BatchNorm2D(32),
                nn.ReLU(),
                # nn.MaxPool2D(2, 2),
                Conv2D(32, 64, 3, padding=1),
                BatchNorm2D(64),
                nn.ReLU(),
                nn.MaxPool2D(2, 2),
            )
            block2_input_filters += 64

        self.block1 = nn.Sequential(
            ZeroPad2D(1),
            Conv2D(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2D(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add_sublayer(name="rpnb1Conv",sublayer=Conv2D(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add_sublayer(name="rpnb1Bn",sublayer=BatchNorm2D(num_filters[0]))
            self.block1.add_sublayer(name="rpnb1Act",sublayer=nn.ReLU())
        self.deconv1 = nn.Sequential(
            Conv2DTranspose(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2D(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            ZeroPad2D(1),
            Conv2D(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2D(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add_sublayer(name="rpnb2Conv",sublayer=
                Conv2D(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add_sublayer(name="rpnb2Bn",sublayer=BatchNorm2D(num_filters[1]))
            self.block2.add_sublayer(name="rpnb2Act",sublayer=nn.ReLU())
        self.deconv2 = nn.Sequential(
            Conv2DTranspose(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2D(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            ZeroPad2D(1),
            Conv2D(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2D(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add_sublayer(name="rpnb3Conv",sublayer=
                Conv2D(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add_sublayer(name="rpnb3Bn",sublayer=BatchNorm2D(num_filters[2]))
            self.block3.add_sublayer(name="rpnb3Act",sublayer=nn.ReLU())
        self.deconv3 = nn.Sequential(
            Conv2DTranspose(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2D(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2D(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2D(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2D(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = paddle.clip(
                paddle.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = paddle.concat([x, self.bev_extractor(bev)], axis=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = paddle.concat([up1, up2, up3], axis=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.transpose((0, 2, 3, 1))
        cls_preds = cls_preds.transpose((0, 2, 3, 1))
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.transpose((0, 2, 3, 1))
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Layer):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="MiddleExtractor",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight

        vfe_class_dict = {
            "VoxelFeatureExtractor": VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": VoxelFeatureExtractorV2,
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        self.voxel_feature_extractor = vfe_class(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance)
        mid_class_dict = {
            "MiddleExtractor": MiddleExtractor,
        }
        mid_class = mid_class_dict[middle_class_name]
        self.middle_feature_extractor = mid_class(
            output_shape,
            use_norm,
            num_input_features=vfe_num_filters[-1],
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)
        if len(middle_num_filters_d2) == 0:
            if len(middle_num_filters_d1) == 0:
                num_rpn_input_filters = vfe_num_filters[-1]
            else:
                num_rpn_input_filters = middle_num_filters_d1[-1]
        else:
            num_rpn_input_filters = middle_num_filters_d2[-1]
        rpn_class_dict = {
            "RPN": RPN,
        }
        rpn_class = rpn_class_dict[rpn_class_name]
        self.rpn = rpn_class(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters * 2,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            axis=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(axis=-1)
        self.rpn_recall = metrics.Recall(axis=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            axis=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", paddle.zeros([1],dtype=paddle.int64))

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        voxel_features = self.voxel_feature_extractor(voxels, num_points)
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size_dev)
        # print('spatial_features',spatial_features.shape)
        if self._use_bev:
            preds_dict = self.rpn(spatial_features, example["bev_map"])
        else:
            preds_dict = self.rpn(spatial_features)
        # preds_dict["voxel_features"] = voxel_features
        # preds_dict["spatial_features"] = spatial_features
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        self._total_forward_time += time.time() - t
        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']

            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            cls_targets = labels * cared.astype(labels.dtype)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'],
                                                   reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].reshape((
                    batch_size_dev, -1, 2))
                weights = (labels > 0).astype(dir_logits.dtype)
                weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(
                    dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight
            else:
                dir_loss = paddle.to_tensor(0)
            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].reshape((batch_size, -1, 7))

        self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].reshape((batch_size, -1))
        batch_imgidx = example['image_idx']

        self._total_forward_time += time.time() - t
        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.reshape((batch_size, -1,
                                               self._box_coder.code_size))
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.reshape((batch_size, -1,
                                               num_class_with_bg))
        batch_box_preds = self._box_coder.decode_paddle(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.reshape((batch_size, -1, 2))
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        # for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
        #         batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
        #         batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        # ):
        for i in range(batch_size): # batch_size = batch_box_preds.shape[0]
            box_preds = batch_box_preds[i,:,:]
            cls_preds = batch_cls_preds[i,:,:]
            if self._use_direction_classifier:
                dir_preds = batch_dir_preds[i,:,:]
            else:
                dir_preds = batch_dir_preds[i]
            rect = batch_rect[i,:,:]
            Trv2c = batch_Trv2c[i,:,:]
            P2 = batch_P2[i,:,:]
            img_idx = batch_imgidx[i]
            a_mask = batch_anchors_mask.astype('int64')[i,:]

            if a_mask is not None:
                # box_preds = box_preds[a_mask] # TODO fix
                # cls_preds = cls_preds[a_mask]
                box_preds = mask_slice_v1(box_preds,a_mask.astype(paddle.bool))
                cls_preds = mask_slice_v1(cls_preds,a_mask.astype(paddle.bool))
            if self._use_direction_classifier:
                if a_mask is not None:
                    # dir_preds = dir_preds[a_mask]
                    dir_preds = mask_slice_v1(dir_preds,a_mask.astype(paddle.bool))
                # print(dir_preds.shape)
                dir_labels = paddle.argmax(dir_preds, axis=-1)
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = paddle.nn.functional.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = paddle.nn.functional.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, axis=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_paddle_ops.rotate_nms
            else:
                nms_func = box_paddle_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = paddle.stack([box_preds[:,i] for i in [0,1,3,4,6]],axis=1)
                if not self._use_rotate_nms:
                    box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_paddle_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_paddle_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            paddle.full([num_dets], i, dtype=paddle.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = paddle.concat(selected_boxes, axis=0)
                    selected_labels = paddle.concat(selected_labels, axis=0)
                    selected_scores = paddle.concat(selected_scores, axis=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = paddle.concat(
                            selected_dir_labels, axis=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = paddle.zeros(
                        [total_scores.shape[0]],
                        dtype=paddle.int64)
                else:
                    top_scores = paddle.max(total_scores, axis=-1)
                    top_labels = paddle.argmax(top_scores, axis=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = paddle.to_tensor(
                        [self._nms_score_threshold]).astype(total_scores.dtype)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        # box_preds = box_preds[top_scores_keep]
                        box_preds = mask_slice_v1(box_preds,top_scores_keep)
                        if self._use_direction_classifier:
                            dir_labels = paddle.masked_select(dir_labels,top_scores_keep)
                        top_labels = paddle.masked_select(top_labels,top_scores_keep)
                    # boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    boxes_for_nms = paddle.stack([box_preds[:,i] for i in [0,1,3,4,6]],axis=1)
                    if not self._use_rotate_nms:
                        box_preds_corners = box_paddle_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_paddle_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0).astype(paddle.uint8) ^ dir_labels.astype(paddle.uint8)
                    box_preds[..., -1] += paddle.where(
                        opp_labels,
                        paddle.to_tensor(np.pi).astype(box_preds.dtype),
                        paddle.to_tensor(0.0).astype(box_preds.dtype))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).astype(box_preds) * np.pi
                if(len(box_preds.shape)==1):
                    box_preds = box_preds.unsqueeze(0)
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_paddle_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_paddle_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_paddle_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = paddle.min(box_corners_in_image, axis=1)
                maxxy = paddle.max(box_corners_in_image, axis=1)
                # minx = paddle.min(box_corners_in_image[..., 0], axis=1)
                # maxx = paddle.max(box_corners_in_image[..., 0], axis=1)
                # miny = paddle.min(box_corners_in_image[..., 1], axis=1)
                # maxy = paddle.max(box_corners_in_image[..., 1], axis=1)
                # box_2d_preds = paddle.stack([minx, miny, maxx, maxy], axis=1)
                box_2d_preds = paddle.concat([minxy, maxxy], axis=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t
        return predictions_dicts

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.astype(paddle.float32)
        self.rpn_metrics.astype(paddle.float32)
        self.rpn_cls_loss.astype(paddle.float32)
        self.rpn_loc_loss.astype(paddle.float32)
        self.rpn_total_loss.astype(paddle.float32)

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = paddle.sin(boxes1[..., -1:]) * paddle.cos(
        boxes2[..., -1:])
    rad_tg_encoding = paddle.cos(boxes1[..., -1:]) * paddle.sin(boxes2[..., -1:])
    boxes1 = paddle.concat([boxes1[..., :-1], rad_pred_encoding], axis=-1)
    boxes2 = paddle.concat([boxes2[..., :-1], rad_tg_encoding], axis=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.reshape((batch_size, -1, box_code_size))
    if encode_background_as_zeros:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class))
    else:
        cls_preds = cls_preds.reshape((batch_size, -1, num_class + 1))
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = paddleplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=paddle.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.astype(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.astype(dtype)
    reg_weights = positives.astype(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.astype(dtype).sum(1, keepdim=True)
        num_examples = paddle.clip(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).astype(dtype)
        reg_weights /= paddle.clip(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).astype(dtype)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = paddle.stack([positives, negatives], axis=-1).astype(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = paddle.clip(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = paddle.clip(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=paddle.float32):
    weights = paddle.zeros(labels.shape, dtype=dtype)
    for label, weight in weight_per_class:
        positives = (labels == label).astype(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = paddle.clip(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.reshape((batch_size, -1, 7))
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).astype(paddle.int64)
    if one_hot:
        dir_cls_targets = paddleplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets

def mask_slice_v1(data, mask):
    """
    data.shape = [x,y], type: float...
    mask.shape = [x] ,type: bool
    in torch, can do it by data[mask] to get shape:[x,y] result.
    """
    data_shape = data.shape
    mask = mask.unsqueeze(-1) # [x,1]
    mask = paddle.tile(mask,[1,data_shape[1]]) # [x,y]
    slice = paddle.masked_select(data,mask) # [x*y]
    return slice.reshape([-1,data_shape[1]]) # [x,y]