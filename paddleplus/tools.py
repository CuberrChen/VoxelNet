import functools
import inspect
import sys
from collections import OrderedDict

import numba
import numpy as np
import paddle


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

def paddle_to_np_dtype(ttype):
    type_map = {
        paddle.float16: np.dtype(np.float16),
        paddle.float32: np.dtype(np.float32),
        paddle.float64: np.dtype(np.float64),
        paddle.int32: np.dtype(np.int32),
        paddle.int64: np.dtype(np.int64),
        paddle.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def einsum(equation, *operands):# get from paddlenlp
    r"""
    Executes the sum of product of provided operands based on the Einstein summation convention.
    Einsum can be used to complete a variety of operations, such as sum, transpose,
    batch matrix multiplication.
    Args:
        equation (`str`):
            Uses uncased letters to specify the dimension of the operands and result. The input
            equation is on the left hand before `->` while the output equation is on the right side.
            Einsum can infer the result shape so that the `->` and the result label letters can be omitted.
            Operands in the input equation are splited by commas (','), e.g. 'abc,cde' describes two 3D
            operands. The dimensions labeled with same letter should be same or be 1. Ellipsis ('...') can
            be used to specify the broadcast dimensions.
        operands (`Tensor`):
            The operands to compute the Einstein sum of.
    Returns:
        `Tensor`: The result of Einstein sum product.
    Example:
    .. code-block::
        import paddle
        import paddlenlp.ops as ops
        x = paddle.randn((4,))
        y = paddle.randn((5,))
        print(ops.einsum('i->', x)) # sum
        print(ops.einsum('i,i->', x, x)) # dot
        print(ops.einsum("i,j->ij", x, y)), # outer
        A = paddle.randn((2, 4, 5))
        B = paddle.randn((2, 5, 6))
        print(ops.einsum('ijk->kji', A)) # transpose
        print(ops.einsum('ijk, ikl->ijl', A,B)) # batch matrix multiplication
        print(ops.einsum('...jk->...kj', A)) # Ellipsis transpose
        print(ops.einsum('...jk, ...kl->...jl', A,B)) # Ellipsis batch matrix multiplication
    """

    def _mul_sum(left, right, sum_dims):
        assert left.rank() == right.rank(), "number of rank should be equal."
        if len(sum_dims) == 0:
            return left * right
        sum_dims_set = set(sum_dims)
        batch_dims = []
        left_out_dims = []
        right_out_dims = []
        batch_size = summed_size = left_size = right_size = 1
        dim = len(left.shape)
        for i in range(dim):
            is_left_summed_dim = left.shape[i] > 1  # not broadcast dim
            is_right_summed_dim = right.shape[i] > 1
            if i in sum_dims_set:
                if is_left_summed_dim and is_right_summed_dim:
                    assert left.shape[i] == right.shape[
                        i], "Non-brocast dim should be equal."
                    summed_size *= left.shape[i]
                elif is_left_summed_dim:
                    left = left.sum(axis=i, keepdim=True)
                elif is_right_summed_dim:
                    right = right.sum(axis=i, keepdim=True)
            elif is_left_summed_dim and is_right_summed_dim:
                assert left.shape[i] == right.shape[
                    i], "Non-brocast dim should be equal."
                batch_dims.append(i)
                batch_size *= left.shape[i]
            elif is_left_summed_dim:
                left_out_dims.append(i)
                left_size *= left.shape[i]
            else:
                right_out_dims.append(i)
                right_size *= right.shape[i]
        out_shape = [left.shape[i] for i in batch_dims + left_out_dims]
        out_shape.extend([1] * len(sum_dims))
        out_shape.extend([right.shape[i] for i in right_out_dims])

        left_perm = list(batch_dims)
        left_perm.extend(left_out_dims)
        left_perm.extend(sum_dims)
        left_perm.extend(right_out_dims)

        right_perm = list(batch_dims)
        right_perm.extend(sum_dims)
        right_perm.extend(right_out_dims)
        right_perm.extend(left_out_dims)

        output_perm = [-1] * (len(batch_dims) + len(left_out_dims) +
                              len(sum_dims) + len(right_out_dims))
        for i, j in enumerate(batch_dims + left_out_dims + sum_dims +
                              right_out_dims):
            output_perm[j] = i

        left = paddle.reshape(
            paddle.transpose(
                left, perm=left_perm), (batch_size, left_size, summed_size))
        right = paddle.reshape(
            paddle.transpose(
                right, perm=right_perm), (batch_size, summed_size, right_size))
        result = paddle.matmul(left, right)
        result = paddle.reshape(result, out_shape)
        result = paddle.transpose(result, output_perm)
        return result

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # Equation is case insensitive
    num_letters = 26
    letters_to_idx = [-1] * num_letters
    equation = equation.lower().replace(' ', '')
    # 1. Parse the equation
    eqns = equation.split("->")
    num_eqns_size = len(eqns)
    assert num_eqns_size <= 2, "The '->' should exist at most only once"

    input_eqn = eqns[0]
    output_eqn = None if num_eqns_size <= 1 else eqns[1]
    operand_eqns = input_eqn.split(",")
    assert len(operand_eqns) == len(
        operands
    ), "Number of operands in equation and the tensors provided should be equal."

    # Parse input equation
    num_total_idxes = 0
    input_operand_idxes = []
    letter_frequence = [0] * num_letters
    idxes_last_operand = []
    num_ell_idxes = -1
    first_ell_idx = 0
    for i, term in enumerate(operand_eqns):
        ell_char_count = 0
        operand_rank = int(operands[i].rank().numpy())
        curr_num_ell_idxes = operand_rank - len(term) + 3
        dims_in_terms = 0
        curr_operand_idxes = []
        for ch in term:
            if ch == '.':
                ell_char_count += 1
                assert ell_char_count <= 3, "The '.' should only exist in one ellispis '...' in term {}".format(
                    term)
                if ell_char_count == 3:
                    if num_ell_idxes == -1:
                        num_ell_idxes = curr_num_ell_idxes
                        first_ell_idx = num_total_idxes
                        num_total_idxes += num_ell_idxes
                    else:
                        assert curr_num_ell_idxes == num_ell_idxes, "Ellispis in all terms should represent same dimensions ({}).".format(
                            num_ell_idxes)

                    for j in range(num_ell_idxes):
                        curr_operand_idxes.append(j + first_ell_idx)
                        idxes_last_operand.append(i)
                    dims_in_terms += num_ell_idxes
            else:
                assert (
                        (ell_char_count == 0) or (ell_char_count == 3)
                ), "'.' must only occur in ellipsis, operand {}".format(term)
                assert (ord('a') <= ord(ch) and
                        ord(ch) <= ord('z')), "only accept alphabet (a-zA-Z)"
                letter_num = ord(ch) - ord('a')
                if letters_to_idx[letter_num] == -1:
                    letters_to_idx[letter_num] = num_total_idxes
                    num_total_idxes += 1
                    idxes_last_operand.append(i)
                else:
                    idxes_last_operand[letters_to_idx[letter_num]] = i
                letter_frequence[letter_num] += 1
                curr_operand_idxes.append(letters_to_idx[letter_num])
                dims_in_terms += 1

        assert dims_in_terms == operand_rank, "Dimension dismatch for operand {}: equation {}, tensor {}".format(
            i, dims_in_terms, operand_rank)
        input_operand_idxes.append(curr_operand_idxes)
    # Parse output equation
    idxes_to_output_dims = [-1] * num_total_idxes
    num_output_dims = 0
    if num_eqns_size == 2:
        ell_char_count = 0
        for ch in output_eqn:
            if ch == '.':
                ell_char_count += 1
                assert ell_char_count <= 3, "The '.' should only exist in one ellispis '...' in term {}".format(
                    output_eqn)
                if ell_char_count == 3:
                    assert num_ell_idxes > -1, "Input equation '{}' don't have ellispis.".format(
                        input_eqn)
                    for j in range(num_ell_idxes):
                        idxes_to_output_dims[first_ell_idx +
                                             j] = num_output_dims
                        num_output_dims += 1

            else:
                assert ((ell_char_count == 0) or (ell_char_count == 3)
                        ), "'.' must only occur in ellipsis, operand {}".format(
                    output_eqn)
                assert (ord('a') <= ord(ch) and
                        ord(ch) <= ord('z')), "only accept alphabet (a-zA-Z)"
                letter_num = ord(ch) - ord('a')
                assert letters_to_idx[
                           letter_num] != -1, "character {} doesn't exist in input".format(
                    ch)
                assert idxes_to_output_dims[letters_to_idx[
                    letter_num]] == -1, "character {} occurs twice in output".format(
                    ch)

                idxes_to_output_dims[letters_to_idx[
                    letter_num]] = num_output_dims
                num_output_dims += 1
    else:  # num_eqns_size == 1
        # Infer the output dims
        if num_ell_idxes >= 0:
            for j in range(num_ell_idxes):
                idxes_to_output_dims[first_ell_idx + j] = num_output_dims
                num_output_dims += 1
        for j in range(num_letters):
            if letter_frequence[j] == 1:
                idxes_to_output_dims[letters_to_idx[j]] = num_output_dims
                num_output_dims += 1

    # Mark sum index
    sum_dim = num_output_dims
    for i in range(num_total_idxes):
        if idxes_to_output_dims[i] == -1:
            idxes_to_output_dims[i] = sum_dim
            sum_dim += 1

    preprocessed_operands = []
    size_dims = [-1] * num_total_idxes
    for i, preprocessed_operand in enumerate(operands):
        idx_to_dims = [-1] * num_total_idxes
        curr_operand_idxes = input_operand_idxes[i]
        dim = 0
        for j, idx in enumerate(curr_operand_idxes):
            output_dim = idxes_to_output_dims[idx]
            if idx_to_dims[output_dim] == -1:
                idx_to_dims[output_dim] = dim
                if size_dims[idx] == -1:
                    size_dims[idx] = preprocessed_operand.shape[dim]
                else:
                    assert size_dims[idx] == preprocessed_operand.shape[
                        dim], "Dimension size does not match previous size. "
                dim += 1
            else:
                # Diagonal repeated index
                # TODO(zhoushunjie): Need to develop a paddle.diagonal api
                raise NotImplementedError("Can't support diagonal.")
        perm = []
        for input_dim in idx_to_dims:
            if input_dim > -1:
                perm.append(input_dim)
        # Transpose the tensor by perm
        preprocessed_operand = paddle.transpose(preprocessed_operand, perm=perm)

        for dim, input_dim in enumerate(idx_to_dims):
            if input_dim == -1:
                preprocessed_operand = paddle.unsqueeze(preprocessed_operand,
                                                        dim)

        preprocessed_operands.append(preprocessed_operand)

    # 2. Execute the mul_sum
    sum_dims = []
    result = preprocessed_operands[0]
    for i in range(num_total_idxes):
        if idxes_last_operand[i] == 0 and idxes_to_output_dims[
            i] >= num_output_dims:
            result = result.sum(axis=idxes_to_output_dims[i], keepdim=True)
    for i in range(1, len(preprocessed_operands)):
        for j in range(num_total_idxes):
            if idxes_last_operand[j] == i and idxes_to_output_dims[
                j] >= num_output_dims:
                sum_dims.append(idxes_to_output_dims[j])
        result = _mul_sum(result, preprocessed_operands[i], sum_dims)

    squeeze_dims = [
        i for i in range(len(result.shape) - 1, num_output_dims - 1, -1)
    ]
    result = paddle.squeeze(result, squeeze_dims)
    return result