#
# SPDX-FileCopyrightText: Copyright 2019-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

file(GLOB SRC_S4 "./*_s4*.c")
file(GLOB SRC_S8 "./*_s8*.c")
file(GLOB SRC_S16 "./*_s16*.c")
set(SRC_FLT "")
if (ARM_NN_ENABLE_F32)
  list(APPEND SRC_FLT arm_convolve_f32.c
                      arm_convolve_1_x_n_f32.c
                      arm_convolve_1x1_f32.c
                      arm_depthwise_conv_f32.c
                      arm_transpose_conv_f32.c)
endif()
if (ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_convolve_f16.c
                      arm_convolve_1_x_n_f16.c
                      arm_convolve_1x1_f16.c
                      arm_depthwise_conv_f16.c
                      arm_transpose_conv_f16.c)
endif()
target_sources(cmsis-nn PRIVATE ${SRC_S4} ${SRC_S8} ${SRC_S16} ${SRC_FLT})
