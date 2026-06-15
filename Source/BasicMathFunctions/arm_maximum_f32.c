/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_maximum_f32.c
 * Description:  Elementwise maximum for float32 tensors
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_minmax_f32_common.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_maximum_f32(const cmsis_nn_context *ctx,
                                    const float32_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims)
{
    return arm_minmax_f32_impl(
        ctx, input_1_data, input_1_dims, input_2_data, input_2_dims, output_data, output_dims, 1);
}
/**
 * @} end of groupElementwise group
 */
