#
# Copyright (c) 2019-2022, 2026 Arm Limited.
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

file(GLOB SRC "./*_s8.c")
file(GLOB SRC_S16 "./*_s16.c")
set(SRC_FLT "")
if (ARM_NN_ENABLE_F32)
  list(APPEND SRC_FLT arm_avg_pool_f32.c
                      arm_max_pool_f32.c)
endif()
if (ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_avg_pool_f16.c
                      arm_max_pool_f16.c)
endif()
target_sources(cmsis-nn PRIVATE ${SRC} ${SRC_S16} ${SRC_FLT})

