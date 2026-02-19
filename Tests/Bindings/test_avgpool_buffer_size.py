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
    get_buffer_size_wrapper_name,
)

from cmsis_nn import avgpool_buffer_size, Backend, DataType

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "dim_dst_width, ch_src",
    [
        (5, 13),
    ],
)
def test_avgpool_buffer_size_matches_raw(dim_dst_width, ch_src):
    if not SHARED_LIB.exists():
        pytest.skip(f"Missing shared CMSIS-NN library at {SHARED_LIB}")

    lib = ctypes.CDLL(str(SHARED_LIB))
    argtypes = [ctypes.c_int32, ctypes.c_int32]

    for backend in Backend.__members__.values():
        for data_type in DataType.__members__.values():
            if data_type == DataType.A8W4:
                continue
            func_name = get_buffer_size_wrapper_name("avgpool", backend, data_type)
            if not func_name:
                raise RuntimeError(f"No raw function mapping for {backend} {data_type}")
            try:
                raw_func = getattr(lib, func_name)
            except AttributeError:
                raise RuntimeError(f"Missing symbol {func_name} in {SHARED_LIB}")

            raw_func.argtypes = argtypes
            raw_func.restype = ctypes.c_int32

            raw = raw_func(dim_dst_width, ch_src)

            py = avgpool_buffer_size(
                backend,
                data_type,
                dim_dst_width=dim_dst_width,
                ch_src=ch_src,
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
