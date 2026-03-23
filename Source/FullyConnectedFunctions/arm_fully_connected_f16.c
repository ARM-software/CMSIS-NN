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
 * Title:        arm_fully_connected_f16.c
 * Description:  Fully connected function (float16)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

static arm_cmsis_nn_status arm_fully_connected_core_f16(const cmsis_nn_context *ctx,
                                                        const cmsis_nn_fc_params_f16 *fc_params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const float16_t *input,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const float16_t *kernel,
                                                        const cmsis_nn_dims *bias_dims,
                                                        const float16_t *bias,
                                                        const cmsis_nn_dims *output_dims,
                                                        float16_t *output)
{
    (void)ctx;
    (void)bias_dims;

    const int32_t batch_cnt = input_dims->n;
    const int32_t input_len = filter_dims->n;
    const int32_t output_len = output_dims->c;

    if (fc_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        return arm_nn_mat_mult_nt_n_packed_f16(input,
                                               kernel,
                                               bias,
                                               output,
                                               batch_cnt,
                                               output_len,
                                               input_len,
                                               output_len,
                                               fc_params->activation.min,
                                               fc_params->activation.max);
    }

    return arm_nn_mat_mult_nt_t_f16(input,
                                    kernel,
                                    bias,
                                    output,
                                    batch_cnt,
                                    output_len,
                                    input_len,
                                    output_len,
                                    fc_params->activation.min,
                                    fc_params->activation.max);
}

/* Refer header file for details. */
arm_cmsis_nn_status arm_fully_connected_nhwc_f16(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_fc_params_f16 *fc_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output)
{
    const int32_t input_len = input_dims->h * input_dims->w * input_dims->c;
    if (filter_dims->n != input_len)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_fully_connected_core_f16(
        ctx, fc_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

/* Refer header file for details. */
arm_cmsis_nn_status arm_fully_connected_f16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params_f16 *fc_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const float16_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const float16_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output,
                                            arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_fully_connected_nhwc_f16(
        ctx, fc_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

/**
 * @} end of FC group
 */

#endif /* ARM_NN_ENABLE_F16 */
