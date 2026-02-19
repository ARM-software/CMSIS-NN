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
import os
import sys
from pathlib import Path

import cmsis_nn

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("CMSIS_NN_BUILD_DIR", ROOT / "build"))
SHARED_LIB = Path(os.environ.get("CMSIS_NN_SHARED_LIB", BUILD_DIR / "libcmsis-nn.so"))

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))


class CmsisNnTile(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_int32),
        ("h", ctypes.c_int32),
    ]


class CmsisNnActivation(ctypes.Structure):
    _fields_ = [
        ("min", ctypes.c_int32),
        ("max", ctypes.c_int32),
    ]


class CmsisNnConvParams(ctypes.Structure):
    _fields_ = [
        ("input_offset", ctypes.c_int32),
        ("output_offset", ctypes.c_int32),
        ("stride", CmsisNnTile),
        ("padding", CmsisNnTile),
        ("dilation", CmsisNnTile),
        ("activation", CmsisNnActivation),
    ]


class CmsisNnDwConvParams(ctypes.Structure):
    _fields_ = [
        ("input_offset", ctypes.c_int32),
        ("output_offset", ctypes.c_int32),
        ("ch_mult", ctypes.c_int32),
        ("stride", CmsisNnTile),
        ("padding", CmsisNnTile),
        ("dilation", CmsisNnTile),
        ("activation", CmsisNnActivation),
    ]


class CmsisNnTransposeConvParams(ctypes.Structure):
    _fields_ = [
        ("input_offset", ctypes.c_int32),
        ("output_offset", ctypes.c_int32),
        ("stride", CmsisNnTile),
        ("padding", CmsisNnTile),
        ("padding_offsets", CmsisNnTile),
        ("dilation", CmsisNnTile),
        ("activation", CmsisNnActivation),
    ]


class CmsisNnDims(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("w", ctypes.c_int32),
        ("c", ctypes.c_int32),
    ]


def make_dims(nhwc):
    return CmsisNnDims(n=nhwc[0], h=nhwc[1], w=nhwc[2], c=nhwc[3])


def make_tile(hw):
    return CmsisNnTile(w=hw[1], h=hw[0])


def make_conv_params(padding_hw, stride_hw, dilation_hw, input_offset, output_offset, activation_min, activation_max):
    return CmsisNnConvParams(
        input_offset=input_offset,
        output_offset=output_offset,
        stride=make_tile(stride_hw),
        padding=make_tile(padding_hw),
        dilation=make_tile(dilation_hw),
        activation=CmsisNnActivation(min=activation_min, max=activation_max),
    )


def make_dw_conv_params(
    padding_hw,
    stride_hw,
    dilation_hw,
    input_offset,
    output_offset,
    ch_mult,
    activation_min,
    activation_max,
):
    return CmsisNnDwConvParams(
        input_offset=input_offset,
        output_offset=output_offset,
        ch_mult=ch_mult,
        stride=make_tile(stride_hw),
        padding=make_tile(padding_hw),
        dilation=make_tile(dilation_hw),
        activation=CmsisNnActivation(min=activation_min, max=activation_max),
    )


def make_transpose_conv_params(
    padding_hw,
    stride_hw,
    dilation_hw,
    padding_offsets_hw,
    input_offset,
    output_offset,
    activation_min,
    activation_max,
):
    return CmsisNnTransposeConvParams(
        input_offset=input_offset,
        output_offset=output_offset,
        stride=make_tile(stride_hw),
        padding=make_tile(padding_hw),
        padding_offsets=make_tile(padding_offsets_hw),
        dilation=make_tile(dilation_hw),
        activation=CmsisNnActivation(min=activation_min, max=activation_max),
    )


BACKEND_SUFFIX = {
    cmsis_nn.Backend.MVE: "mve",
    cmsis_nn.Backend.DSP: "dsp",
    cmsis_nn.Backend.SCALAR: "",
}

DATATYPE_PREFIX = {
    cmsis_nn.DataType.A8W4: "s4",
    cmsis_nn.DataType.A8W8: "s8",
    cmsis_nn.DataType.A16W8: "s16",
}


def get_buffer_size_wrapper_name(op_name, backend, data_type):
    prefix = DATATYPE_PREFIX.get(data_type)
    if prefix is None:
        return None
    suffix = BACKEND_SUFFIX.get(backend)
    if suffix is None:
        return None
    if suffix:
        return f"arm_{op_name}_{prefix}_get_buffer_size_{suffix}"
    return f"arm_{op_name}_{prefix}_get_buffer_size"
