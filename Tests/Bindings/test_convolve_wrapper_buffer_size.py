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
    CmsisNnConvParams,
    CmsisNnDims,
    get_buffer_size_wrapper_name,
    make_conv_params,
    make_dims,
)

import cmsis_nn
from cmsis_nn import convolve_wrapper_buffer_size, Backend, DataType, FilterFormat

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("filter_format", [None, FilterFormat.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_nhwc, filter_nhwc, output_nhwc, padding_hw, stride_hw, dilation_hw, input_offset, output_offset, activation_min, activation_max",
    [
        ((1, 8, 8, 16), (8, 3, 3, 16), (1, 6, 6, 8), (0, 0), (1, 1), (1, 1), 0, 0, -128, 127),
    ],
)
def test_convolve_wrapper_buffer_size_matches_raw(
    filter_format,
    input_nhwc,
    filter_nhwc,
    output_nhwc,
    padding_hw,
    stride_hw,
    dilation_hw,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    argtypes = [
        ctypes.POINTER(CmsisNnConvParams),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
    ]

    conv_params = make_conv_params(
        padding_hw, stride_hw, dilation_hw, input_offset, output_offset, activation_min, activation_max
    )
    input_dims = make_dims(input_nhwc)
    filter_dims = make_dims(filter_nhwc)
    output_dims = make_dims(output_nhwc)

    for backend in Backend.__members__.values():
        for data_type in DataType.__members__.values():
            func_name = get_buffer_size_wrapper_name("convolve_wrapper", backend, data_type)
            if not func_name:
                raise RuntimeError(f"No raw function mapping for {backend} {data_type}")
            try:
                raw_func = getattr(lib, func_name)
            except AttributeError:
                raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

            raw_func.argtypes = argtypes
            raw_func.restype = ctypes.c_int32

            raw = raw_func(
                ctypes.byref(conv_params),
                ctypes.byref(input_dims),
                ctypes.byref(filter_dims),
                ctypes.byref(output_dims),
            )

            py = convolve_wrapper_buffer_size(
                backend,
                data_type,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                output_nhwc=output_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                input_offset=input_offset,
                output_offset=output_offset,
                activation_min=activation_min,
                activation_max=activation_max,
                **({} if filter_format is None else {"filter_format": filter_format}),
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


@pytest.mark.skipif(not cmsis_nn.aot_enabled, reason="Packed filters require AOT")
@pytest.mark.parametrize("filter_format", [FilterFormat.ROW_MAJOR, FilterFormat.PACKED_4X16])
@pytest.mark.parametrize(
    "filter_nhwc, output_nhwc, padding_hw, stride_hw, dilation_hw, packed_size",
    [
        ((7, 1, 1, 8), (1, 1, 9, 7), (0, 0), (1, 1), (1, 1), 0),
        ((7, 1, 3, 8), (1, 1, 9, 7), (0, 1), (1, 1), (1, 1), 128),
        ((7, 1, 3, 8), (1, 1, 5, 7), (0, 1), (1, 2), (1, 1), 128),
        ((7, 1, 3, 8), (1, 1, 9, 7), (0, 2), (1, 1), (1, 2), 128),
    ],
)
def test_mve_filter_format_matches_raw(
    filter_format, filter_nhwc, output_nhwc, padding_hw, stride_hw, dilation_hw, packed_size
):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")
    lib = ctypes.CDLL(str(SHARED_LIB))
    raw_func = lib.arm_convolve_wrapper_s8_get_buffer_size_mve
    raw_func.argtypes = [ctypes.POINTER(CmsisNnConvParams)] + [ctypes.POINTER(CmsisNnDims)] * 3
    raw_func.restype = ctypes.c_int32
    conv_params = make_conv_params(padding_hw, stride_hw, dilation_hw, 0, 0, -128, 127)
    conv_params.filter_format = filter_format.value
    input_dims = make_dims((1, 1, 9, 8))
    filter_dims = make_dims(filter_nhwc)
    output_dims = make_dims(output_nhwc)
    expected = raw_func(
        ctypes.byref(conv_params), ctypes.byref(input_dims), ctypes.byref(filter_dims), ctypes.byref(output_dims)
    )
    size = convolve_wrapper_buffer_size(
        Backend.MVE,
        DataType.A8W8,
        input_nhwc=(1, 1, 9, 8),
        filter_nhwc=filter_nhwc,
        output_nhwc=output_nhwc,
        padding_hw=padding_hw,
        stride_hw=stride_hw,
        dilation_hw=dilation_hw,
        filter_format=filter_format,
    )
    assert size == expected
    if filter_format == FilterFormat.PACKED_4X16:
        assert size == packed_size


@pytest.mark.skipif(cmsis_nn.aot_enabled, reason="Requires an AOT-disabled build")
def test_packed_filter_requires_aot():
    with pytest.raises(ValueError, match="ARM_NN_ENABLE_AOT=ON"):
        convolve_wrapper_buffer_size(
            Backend.MVE,
            DataType.A8W8,
            input_nhwc=(1, 1, 9, 8),
            filter_nhwc=(7, 1, 3, 8),
            output_nhwc=(1, 1, 9, 7),
            padding_hw=(0, 1),
            stride_hw=(1, 1),
            dilation_hw=(1, 1),
            filter_format=FilterFormat.PACKED_4X16,
        )
