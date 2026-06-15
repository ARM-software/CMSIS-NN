/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_conv1x1_opt_f16.h
 * Description:  Float16 conv1x1 specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV1X1_OPT_F16_H
#define ARM_CONV1X1_OPT_F16_H

/* Internal specialization helpers (included from arm_convolve_1x1_f16.c). */

#include "Internal/arm_conv1x1_opt_common.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION
typedef bool (*arm_conv1x1_match_f16)(const cmsis_nn_context *ctx,
                                      const cmsis_nn_conv_params_f16 *params,
                                      const cmsis_nn_dims *input_dims,
                                      const float16_t *input_data,
                                      const cmsis_nn_dims *filter_dims,
                                      const float16_t *filter_data,
                                      const cmsis_nn_dims *bias_dims,
                                      const float16_t *bias_data,
                                      const cmsis_nn_dims *output_dims,
                                      float16_t *output_data);

typedef arm_cmsis_nn_status (*arm_conv1x1_call_f16)(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_conv_params_f16 *params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const float16_t *input_data,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const float16_t *filter_data,
                                                    const cmsis_nn_dims *bias_dims,
                                                    const float16_t *bias_data,
                                                    const cmsis_nn_dims *output_dims,
                                                    float16_t *output_data);

typedef struct
{
    arm_conv1x1_match_f16 match;
    arm_conv1x1_call_f16 call;
} arm_conv1x1_spec_f16;
#endif

#endif /* ARM_CONV1X1_OPT_F16_H */
