# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import io

import cmsis_nn

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not cmsis_nn.aot_enabled, reason="AOT disabled")

from .test_bindings_common import SHARED_LIB, CmsisNnDims


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("k", [1, 7, 15, 16, 17, 31, 32, 33])
def test_layout_and_padding(groups, channels, k):
    weights = (np.arange(groups * channels * k) % 251 - 125).astype(np.int8).reshape(groups * channels, k)
    actual = cmsis_nn.convolve_s8_pack_weights(weights, groups=groups)
    padded = np.zeros((groups, (channels + 3) // 4 * 4, (k + 15) // 16 * 16), dtype=np.int8)
    padded[:, :channels, :k] = weights.reshape(groups, channels, k)
    expected = padded.reshape(groups, (channels + 3) // 4, 4, (k + 15) // 16, 16).transpose(0, 1, 3, 2, 4)
    np.testing.assert_array_equal(actual, expected.flatten())
    assert actual.dtype == np.int8 and actual.flags.c_contiguous
    assert actual.strides == (1,)
    assert actual.shape == (actual.size,)
    assert actual.size == cmsis_nn.convolve_s8_packed_filter_size([groups * channels, 1, 1, k], groups)


@pytest.mark.parametrize("shape,groups", [((5, 17), 1), ((6, 2, 3, 5), 2), ((9, 3, 2, 1), 3)])
@pytest.mark.parametrize("offset", [-127, 0, 128])
@pytest.mark.parametrize("with_bias", [False, True])
def test_preparation_and_serialization(shape, groups, offset, with_bias):
    weights = (np.arange(np.prod(shape)) % 29 - 14).astype(np.int8).reshape(shape)
    bias = np.arange(shape[0], dtype=np.int32) * 139 if with_bias else None
    packed, effective = cmsis_nn.convolve_s8_prepare_weights(weights, offset, bias, groups=groups)
    expected = offset * weights.reshape(shape[0], -1).astype(np.int32).sum(axis=1, dtype=np.int32)
    if bias is not None:
        expected += bias
    np.testing.assert_array_equal(effective, expected)
    assert effective.shape == (shape[0],) and effective.dtype == np.int32
    assert effective.strides == (4,) and effective.flags.c_contiguous
    assert packed.strides == (1,) and packed.flags.c_contiguous
    stream = io.BytesIO()
    np.savez(stream, packed_weights=packed, effective_bias=effective)
    stream.seek(0)
    with np.load(stream) as saved:
        np.testing.assert_array_equal(saved["packed_weights"], packed)
        np.testing.assert_array_equal(saved["effective_bias"], effective)
    np.testing.assert_array_equal(np.frombuffer(packed.tobytes(), dtype=np.int8), packed)
    np.testing.assert_array_equal(cmsis_nn.convolve_s8_pack_weights(weights.reshape(shape[0], -1), groups), packed)


@pytest.mark.parametrize("dims,groups", [
    ([0, 1, 1, 1], 1), ([1, 0, 1, 1], 1), ([1, 1, -1, 1], 1), ([1, 1, 1, 0], 1),
    ([6, 2, 1, 9], 0), ([6, 2, 1, 9], -1), ([6, 2, 1, 9], 4),
    ([2**31 - 1] * 4, 1), ([1, 1, 1, 2**31 - 1], 1),
    ([2**31 - 1, 1, 1, 1], 2**31 - 1), ([1, 65536, 65536, 1], 1),
])
def test_invalid_sizes(dims, groups):
    assert cmsis_nn.convolve_s8_packed_filter_size(dims, groups) == 0


@pytest.mark.parametrize("weights,error", [
    (np.ones((2, 3), dtype=np.int16), TypeError),
    (np.ones((2, 3), dtype=np.float32), TypeError),
    (np.array(1, dtype=np.int8), ValueError),
    (np.ones((2, 3, 4), dtype=np.int8), ValueError),
    (np.empty((2, 0), dtype=np.int8), ValueError),
    (np.ones((2, 6), dtype=np.int8)[:, ::2], ValueError),
])
def test_invalid_weights(weights, error):
    with pytest.raises(error):
        cmsis_nn.convolve_s8_pack_weights(weights)
    with pytest.raises(error):
        cmsis_nn.convolve_s8_prepare_weights(weights, 13)


@pytest.mark.parametrize("groups", [0, -1, 4])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        cmsis_nn.convolve_s8_prepare_weights(np.ones((6, 17), dtype=np.int8), 13, groups=groups)


@pytest.mark.parametrize("bias,error", [
    (np.ones(6, dtype=np.int16), TypeError), (np.ones((6, 1), dtype=np.int32), ValueError),
    (np.ones(5, dtype=np.int32), ValueError), (np.arange(12, dtype=np.int32)[::2], ValueError),
])
def test_invalid_bias(bias, error):
    with pytest.raises(error):
        cmsis_nn.convolve_s8_prepare_weights(np.ones((6, 17), dtype=np.int8), -13, bias, groups=2)


def test_c_packing_validation():
    lib = ctypes.CDLL(str(SHARED_LIB))
    pack = lib.arm_convolve_s8_pack_weights
    pack.argtypes = [ctypes.POINTER(CmsisNnDims), ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    pack.restype = ctypes.c_int
    size = lib.arm_convolve_s8_packed_get_filter_size
    size.argtypes = [ctypes.POINTER(CmsisNnDims), ctypes.c_int32]
    size.restype = ctypes.c_int32
    dims = CmsisNnDims(6, 2, 1, 9)
    buffer = ctypes.create_string_buffer(600)
    base = ctypes.addressof(buffer)
    assert size(None, 1) == 0
    assert size(ctypes.byref(dims), 2) == 256
    for src, dst in [(None, base), (base, None), (base, base), (base, base + 107), (base + 255, base)]:
        assert pack(ctypes.byref(dims), src, 2, dst) == -1
    assert pack(None, base, 2, base + 300) == -1
    assert pack(ctypes.byref(dims), base, 4, base + 300) == -1
    assert pack(ctypes.byref(dims), base, 2, base + 108) == 0
    assert pack(ctypes.byref(dims), base + 256, 2, base) == 0



@pytest.mark.parametrize("entry", [
    "arm_convolve_s8", "arm_convolve_wrapper_s8", "arm_convolve_1x1_s8",
    "arm_convolve_1x1_s8_fast", "arm_convolve_1_x_n_s8",
])
@pytest.mark.parametrize("formats,bias_present,expected", [
    ((0, 1), True, -1), ((1, 0), True, -1), ((2, 1), True, -1),
    ((1, 2), True, -1), ((-1, -1), True, -1),
    ((1, 1), False, -1), ((1, 1), True, -2),
])
def test_inference_format_validation(entry, formats, bias_present, expected):
    from .test_bindings_common import CmsisNnConvParams

    lib = ctypes.CDLL(str(SHARED_LIB))
    params = CmsisNnConvParams()
    params.filter_format, params.bias_format = formats
    shape = CmsisNnDims(1, 1, 1, 1)
    bias = ctypes.c_int32(0)
    func = getattr(lib, entry)
    # Valid dimensions let the wrapper select a route. All data and scratch
    # pointers are null to prove validation precedes access to packed data.
    args = [None, ctypes.byref(params), None, ctypes.byref(shape), None,
            ctypes.byref(shape), None, None,
            ctypes.byref(bias) if bias_present else None]
    if entry == "arm_convolve_s8":
        args.append(None)
    args += [ctypes.byref(shape), None]
    func.argtypes = [ctypes.c_void_p] * len(args)
    func.restype = ctypes.c_int32
    assert func(*args) == expected


def test_1xn_scratch_routing():
    from .test_bindings_common import CmsisNnConvParams

    lib = ctypes.CDLL(str(SHARED_LIB))
    params = CmsisNnConvParams()
    params.stride.w = params.stride.h = 1
    params.dilation.w = params.dilation.h = 1
    params.padding.w = 1
    input_dims = CmsisNnDims(1, 1, 8, 4)
    filter_dims = CmsisNnDims(3, 1, 3, 4)
    output_dims = CmsisNnDims(1, 1, 8, 3)
    args = [ctypes.byref(params), ctypes.byref(input_dims),
            ctypes.byref(filter_dims), ctypes.byref(output_dims)]
    names = ['arm_convolve_1_x_n_s8_get_buffer_size',
             'arm_convolve_wrapper_s8_get_buffer_size',
             'arm_convolve_wrapper_s8_get_buffer_size_mve',
             'arm_convolve_wrapper_s8_get_buffer_size_dsp']
    for name in names:
        getattr(lib, name).argtypes = [ctypes.c_void_p] * 4
        getattr(lib, name).restype = ctypes.c_int32
    legacy_direct = getattr(lib, names[0])(*args)
    assert getattr(lib, names[2])(*args) == 16
    params.filter_format = params.bias_format = 1
    assert getattr(lib, names[0])(*args) == legacy_direct
    assert getattr(lib, names[2])(*args) == 64
    general = lib.arm_convolve_s8_get_buffer_size
    general.argtypes = [ctypes.c_void_p] * 2
    general.restype = ctypes.c_int32
    expected = general(ctypes.byref(input_dims), ctypes.byref(filter_dims))
    assert getattr(lib, names[1])(*args) == expected
    assert getattr(lib, names[3])(*args) == expected


@pytest.mark.parametrize('formats,bias_present,expected', [
    ((0, 1), True, -1), ((1, 0), True, -1), ((2, 1), True, -1),
    ((1, 2), True, -1), ((-1, -1), True, -1),
    ((1, 1), False, -1), ((1, 1), True, -2),
])
def test_direct_1xn_rejects_before_tensor_access(formats, bias_present, expected):
    from .test_bindings_common import CmsisNnConvParams

    lib = ctypes.CDLL(str(SHARED_LIB))
    params = CmsisNnConvParams()
    params.filter_format, params.bias_format = formats
    bias = ctypes.c_int32(0)
    func = lib.arm_convolve_1_x_n_s8
    func.argtypes = [ctypes.c_void_p] * 11
    func.restype = ctypes.c_int32
    assert func(None, ctypes.byref(params), None, None, None, None, None, None,
                ctypes.byref(bias) if bias_present else None, None, None) == expected
