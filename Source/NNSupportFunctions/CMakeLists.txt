#
# SPDX-FileCopyrightText: Copyright 2019-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
file(GLOB SRC_S32 "./*_s32*.c")
set(SRC_FLT "")
if (ARM_NN_ENABLE_F32 OR ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_nntables_flt.c)
endif()
if (ARM_NN_ENABLE_F32)
  list(APPEND SRC_FLT arm_batch_norm_f32.c
                      arm_get_buffer_size_f32.c
                      arm_nn_conv1d_k3_f32.c
                      arm_nn_conv1d_k3_packed_f32.c
                      arm_nn_conv1d_k5_f32.c
                      arm_nn_conv1d_k5_packed_f32.c
                      arm_nn_depthwise_conv1d_k3_f32.c
                      arm_nn_depthwise_conv3x3_f32.c
                      arm_nn_depthwise_conv_nt_t_f32.c
                      arm_nn_maxpool1d_f32.c
                      arm_nn_pack_conv_patch_f32.c
                      arm_nn_lstm_step_f32.c)
endif()
if (ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_batch_norm_f16.c
                      arm_get_buffer_size_f16.c
                      arm_nn_conv1d_k3_f16.c
                      arm_nn_conv1d_k3_packed_f16.c
                      arm_nn_conv1d_k5_f16.c
                      arm_nn_conv1d_k5_packed_f16.c
                      arm_nn_depthwise_conv1d_k3_f16.c
                      arm_nn_depthwise_conv2x5_f16.c
                      arm_nn_depthwise_conv3x3_f16.c
                      arm_nn_depthwise_conv_nt_t_f16.c
                      arm_nn_pack_conv_patch_f16.c
                      arm_nn_lstm_step_f16.c
                      arm_nn_maxpool1d_f16.c)
endif()
if (ARM_NN_ENABLE_F32)
  list(APPEND SRC_FLT arm_nn_mat_mult_nt_t_f32.c
                      arm_nn_mat_mult_nt_n_packed_f32.c)
endif()
if (ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_nn_mat_mult_nt_t_f16.c
                      arm_nn_mat_mult_nt_n_packed_f16.c)
endif()

target_sources(cmsis-nn PRIVATE ${SRC_S4} ${SRC_S8} ${SRC_S16} ${SRC_S32} ${SRC_FLT} arm_nntables.c
  arm_q7_to_q15_with_offset.c
  arm_s8_to_s16_unordered_with_offset.c)
