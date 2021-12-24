import paddle


class GroupNorm(paddle.nn.GroupNorm):
    def __init__(self, num_channels, num_groups, epsilon=1e-5, affine=True):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            epsilon=epsilon,
            affine=affine)
