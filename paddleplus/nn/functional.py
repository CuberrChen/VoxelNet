import paddle

# def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
#     tensor_onehot = torch.zeros((
#         *list(tensor.shape), depth), dtype=dtype)
#     tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
#     return tensor_onehot

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=paddle.float32):
    return paddle.nn.functional.one_hot(tensor.astype(paddle.int64), depth).astype(dtype)

# if __name__ == '__main__':
#     input = paddle.rand([2,10])
#     intput_t = torch.from_numpy(input.numpy())
#     a = one_hot_f(input,depth=5)
#     b = one_hot(intput_t,depth=5)
#     print(a.numpy()-b.numpy())