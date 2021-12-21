from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from voxelnet.core.box_coders import GroundBox3dCoder, BevBoxCoder
from voxelnet.pypaddle.core import box_paddle_ops
import paddle

class GroundBox3dCoderPaddle(GroundBox3dCoder):
    def encode_paddle(self, boxes, anchors):
        return box_paddle_ops.voxelnet_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_paddle(self, boxes, anchors):
        return box_paddle_ops.voxelnet_box_decode(boxes, anchors, self.vec_encode, self.linear_dim)



class BevBoxCoderPaddle(BevBoxCoder):
    def encode_paddle(self, boxes, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        boxes = boxes[..., [0, 1, 3, 4, 6]]
        return box_paddle_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_paddle(self, encodings, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = box_paddle_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
        z_fixed = paddle.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
        h_fixed = paddle.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
        return paddle.concat([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], axis=-1)

