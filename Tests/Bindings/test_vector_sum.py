# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import cmsis_nn


def reference(weights, lhs_offset, bias=None, rhs_offset=0):
    rows = weights.reshape(weights.shape[0], -1).astype(np.int32)
    result = lhs_offset * (rows.sum(axis=1, dtype=np.int32) + rows.shape[1] * rhs_offset)
    if bias is not None:
        result = result + bias
    return result.astype(np.int32)


@pytest.mark.parametrize("shape", [(4,), (3, 5), (2, 3, 2, 5)])
def test_vector_sum_s8(shape):
    weights = np.arange(np.prod(shape), dtype=np.int16).reshape(shape).astype(np.int8) - 7
    actual = cmsis_nn.vector_sum_s8(weights, -13)
    np.testing.assert_array_equal(actual, reference(weights, -13))
    assert actual.dtype == np.int32
    assert actual.strides == (4,) and actual.flags.c_contiguous


def test_vector_sum_s8_bias_and_offsets():
    weights = np.array([[[[1, -2], [3, 4]]], [[[-5, 6], [7, -8]]]], dtype=np.int8)
    bias = np.array([1234, -5678], dtype=np.int32)
    actual = cmsis_nn.vector_sum_s8(weights, 17, bias=bias, rhs_offset=-9)
    np.testing.assert_array_equal(actual, reference(weights, 17, bias, -9))


@pytest.mark.parametrize("weights", [np.array(3, dtype=np.int8), np.empty((0, 3), dtype=np.int8), np.empty((2, 0), dtype=np.int8)])
def test_vector_sum_s8_rejects_invalid_shape(weights):
    with pytest.raises(ValueError):
        cmsis_nn.vector_sum_s8(weights, 1)


def test_vector_sum_s8_rejects_wrong_weight_dtype():
    with pytest.raises(TypeError):
        cmsis_nn.vector_sum_s8(np.ones((2, 3), dtype=np.int16), 1)


def test_vector_sum_s8_rejects_non_contiguous_weights():
    weights = np.arange(24, dtype=np.int8).reshape(4, 6)[:, ::2]
    assert not weights.flags.c_contiguous
    with pytest.raises(ValueError):
        cmsis_nn.vector_sum_s8(weights, 1)


@pytest.mark.parametrize("bias", [np.ones(2, dtype=np.int16), np.ones((2, 1), dtype=np.int32), np.ones(3, dtype=np.int32)])
def test_vector_sum_s8_rejects_invalid_bias(bias):
    error = TypeError if bias.dtype != np.int32 else ValueError
    with pytest.raises(error):
        cmsis_nn.vector_sum_s8(np.ones((2, 3), dtype=np.int8), 1, bias=bias)


def test_vector_sum_s8_rejects_non_contiguous_bias():
    bias = np.arange(4, dtype=np.int32)[::2]
    with pytest.raises(ValueError):
        cmsis_nn.vector_sum_s8(np.ones((2, 3), dtype=np.int8), 1, bias=bias)


def test_vector_sum_exact_bias_values():
    actual = cmsis_nn.vector_sum_s8(np.zeros((4, 1), dtype=np.int8), 0,
                                    bias=np.array([10, 20, 30, 40], dtype=np.int32))
    np.testing.assert_array_equal(actual, [10, 20, 30, 40])
    assert actual.shape == (4,) and actual.dtype == np.int32
    assert actual.strides == (4,) and actual.flags.c_contiguous
