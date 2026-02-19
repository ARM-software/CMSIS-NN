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
    get_buffer_size_wrapper_name,
    make_dims,
)

from cmsis_nn import svdf_buffer_size, Backend, DataType

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "filter_nhwc",
    [
        (4, 1, 1, 8),
    ],
)
def test_svdf_buffer_size_matches_raw(filter_nhwc):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    argtypes = [ctypes.POINTER(CmsisNnDims)]
    filter_dims = make_dims(filter_nhwc)

    for backend in Backend.__members__.values():
        for data_type in DataType.__members__.values():
            if data_type != DataType.A8W8:
                continue
            func_name = get_buffer_size_wrapper_name("svdf", backend, data_type)
            if not func_name:
                raise RuntimeError(f"No raw function mapping for {backend} {data_type}")
            try:
                raw_func = getattr(lib, func_name)
            except AttributeError:
                raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

            raw_func.argtypes = argtypes
            raw_func.restype = ctypes.c_int32

            raw = raw_func(ctypes.byref(filter_dims))

            py = svdf_buffer_size(
                backend,
                data_type,
                filter_nhwc=filter_nhwc,
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
