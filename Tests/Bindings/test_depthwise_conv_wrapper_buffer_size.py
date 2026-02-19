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
    CmsisNnDwConvParams,
    get_buffer_size_wrapper_name,
    make_dims,
    make_dw_conv_params,
)

from cmsis_nn import depthwise_conv_wrapper_buffer_size, Backend, DataType

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_nhwc, ch_mult, padding_hw, stride_hw, dilation_hw, input_offset, output_offset, activation_min, activation_max",
    [
        ((1, 8, 8, 16), 1, (0, 0), (1, 1), (1, 1), 0, 0, -128, 127),
    ],
)
def test_depthwise_conv_wrapper_buffer_size_matches_raw(
    input_nhwc,
    ch_mult,
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
        ctypes.POINTER(CmsisNnDwConvParams),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
        ctypes.POINTER(CmsisNnDims),
    ]
    filter_nhwc = (1, 3, 3, input_nhwc[3] * ch_mult)
    output_nhwc = (1, 6, 6, input_nhwc[3] * ch_mult)

    dw_conv_params = make_dw_conv_params(
        padding_hw,
        stride_hw,
        dilation_hw,
        input_offset,
        output_offset,
        ch_mult,
        activation_min,
        activation_max,
    )
    input_dims = make_dims(input_nhwc)
    filter_dims = make_dims(filter_nhwc)
    output_dims = make_dims(output_nhwc)

    for backend in Backend.__members__.values():
        for data_type in DataType.__members__.values():
            func_name = get_buffer_size_wrapper_name("depthwise_conv_wrapper", backend, data_type)
            if not func_name:
                raise RuntimeError(f"No raw function mapping for {backend} {data_type}")
            try:
                raw_func = getattr(lib, func_name)
            except AttributeError:
                raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

            raw_func.argtypes = argtypes
            raw_func.restype = ctypes.c_int32

            raw = raw_func(
                ctypes.byref(dw_conv_params),
                ctypes.byref(input_dims),
                ctypes.byref(filter_dims),
                ctypes.byref(output_dims),
            )

            py = depthwise_conv_wrapper_buffer_size(
                backend,
                data_type,
                input_nhwc=input_nhwc,
                filter_nhwc=filter_nhwc,
                output_nhwc=output_nhwc,
                padding_hw=padding_hw,
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                ch_mult=ch_mult,
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
