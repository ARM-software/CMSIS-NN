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
#

import ctypes
import logging

import pytest

from .test_bindings_common import (
    SHARED_LIB,
    CmsisNnDims,
    CmsisNnTransposeConvParams,
    make_dims,
    make_transpose_conv_params,
)

from cmsis_nn import (
    transpose_conv_buffer_size,
    transpose_conv_reverse_conv_buffer_size,
    Backend,
    DataType,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_nhwc, filter_nhwc, output_nhwc, padding_hw, stride_hw, dilation_hw, padding_offsets_hw, input_offset, output_offset, activation_min, activation_max",
    [
        ((1, 4, 4, 8), (4, 3, 3, 8), (1, 6, 6, 4), (0, 0), (1, 1), (1, 1), (0, 0), 0, 0, -128, 127),
    ],
)
def test_transpose_conv_buffer_size_matches_raw(
    input_nhwc,
    filter_nhwc,
    output_nhwc,
    padding_hw,
    stride_hw,
    dilation_hw,
    padding_offsets_hw,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    scalar_argtypes = [
        ctypes.POINTER(CmsisNnTransposeConvParams),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
    ]
    func_names = {
        Backend.MVE: "arm_transpose_conv_s8_get_buffer_size_mve",
        Backend.DSP: "arm_transpose_conv_s8_get_buffer_size",
        Backend.SCALAR: "arm_transpose_conv_s8_get_buffer_size",
    }

    params = make_transpose_conv_params(
        padding_hw,
        stride_hw,
        dilation_hw,
        padding_offsets_hw,
        input_offset,
        output_offset,
        activation_min,
        activation_max,
    )
    input_dims = make_dims(input_nhwc)
    filter_dims = make_dims(filter_nhwc)
    output_dims = make_dims(output_nhwc)

    for backend in Backend.__members__.values():
        data_type = DataType.A8W8
        func_name = func_names[backend]
        try:
            raw_func = getattr(lib, func_name)
        except AttributeError:
            raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

        raw_func.argtypes = scalar_argtypes
        raw_func.restype = ctypes.c_int32
        raw = raw_func(
            ctypes.byref(params),
            ctypes.byref(input_dims),
            ctypes.byref(filter_dims),
            ctypes.byref(output_dims),
        )

        py = transpose_conv_buffer_size(
            backend,
            data_type,
            input_nhwc=input_nhwc,
            filter_nhwc=filter_nhwc,
            output_nhwc=output_nhwc,
            padding_hw=padding_hw,
            stride_hw=stride_hw,
            dilation_hw=dilation_hw,
            padding_offsets_hw=padding_offsets_hw,
            input_offset=input_offset,
            output_offset=output_offset,
            activation_min=activation_min,
            activation_max=activation_max,
        )
        logger.debug(
            "Comparing raw C func %s with python buffer size=%d raw buffer size=%d (%s %s)",
            func_name,
            py,
            raw,
            backend,
            data_type,
        )
        assert py == raw

@pytest.mark.parametrize(
    "input_nhwc, filter_nhwc, stride_hw, expected",
    [
        ((1, 4, 4, 8), (4, 3, 3, 8), (1, 1), 0),
        ((1, 4, 4, 17), (4, 3, 3, 17), (2, 2), 17 * 3 * 3 * 4),
    ],
)
def test_transpose_conv_reverse_conv_buffer_size_matches_raw(input_nhwc, filter_nhwc, stride_hw, expected):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    raw_func_name = "arm_transpose_conv_s8_get_reverse_conv_buffer_size"
    argtypes = [
        ctypes.POINTER(CmsisNnTransposeConvParams),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
    ]

    padding_hw = (0, 0)
    dilation_hw = (1, 1)
    padding_offsets_hw = (0, 0)
    input_offset = 0
    output_offset = 0
    activation_min = -128
    activation_max = 127

    params = make_transpose_conv_params(
        padding_hw,
        stride_hw,
        dilation_hw,
        padding_offsets_hw,
        input_offset,
        output_offset,
        activation_min,
        activation_max,
    )
    input_dims = make_dims(input_nhwc)
    filter_dims = make_dims(filter_nhwc)

    try:
        raw_func = getattr(lib, raw_func_name)
    except AttributeError:
        raise RuntimeError(f"Missing symbol {raw_func_name} in {SHARED_LIB}")

    raw_func.argtypes = argtypes
    raw_func.restype = ctypes.c_int32

    raw = raw_func(
        ctypes.byref(params),
        ctypes.byref(input_dims),
        ctypes.byref(filter_dims),
    )

    for backend in Backend.__members__.values():
        py = transpose_conv_reverse_conv_buffer_size(
            backend,
            DataType.A8W8,
            input_nhwc=input_nhwc,
            filter_nhwc=filter_nhwc,
            padding_hw=padding_hw,
            stride_hw=stride_hw,
            dilation_hw=dilation_hw,
            padding_offsets_hw=padding_offsets_hw,
            input_offset=input_offset,
            output_offset=output_offset,
            activation_min=activation_min,
            activation_max=activation_max,
        )
        logger.debug(
            "Comparing raw C func %s with python reverse buffer size=%d raw buffer size=%d (%s %s)",
            raw_func_name,
            py,
            raw,
            backend,
            DataType.A8W8,
        )
        assert py == raw
        assert py == expected
