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
 * Title:        arm_avg_pool_f32.c
 * Description:  Avg pooling function implementations (float32)
 *
 * $Date:        2 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Pooling
 * @{
 */

/*
 * Avg pooling function for f32
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_avg_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst)
{
    (void)ctx;
    if (!pool_params || !input_dims || !filter_dims || !output_dims || !src || !dst)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const float32_t act_min = pool_params->activation.min;
    const float32_t act_max = pool_params->activation.max;
    const int32_t channel_in = input_dims->c;
    const int32_t batch_size = input_x * input_y * channel_in;
    int32_t batch_cnt = input_dims->n;

    if (batch_cnt < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    while (batch_cnt)
    {
        for (int32_t i_y = 0, base_idx_y = -pad_y; i_y < output_y; base_idx_y += stride_y, i_y++)
        {
            for (int32_t i_x = 0, base_idx_x = -pad_x; i_x < output_x; base_idx_x += stride_x, i_x++)
            {
                const int32_t ker_y_start = MAX(0, -base_idx_y);
                const int32_t ker_x_start = MAX(0, -base_idx_x);
                const int32_t kernel_y_end = MIN(kernel_y, input_y - base_idx_y);
                const int32_t kernel_x_end = MIN(kernel_x, input_x - base_idx_x);
                const int32_t count = (kernel_y_end - ker_y_start) * (kernel_x_end - ker_x_start);

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                const float32x4_t v_act_min = vdupq_n_f32(act_min);
                const float32x4_t v_act_max = vdupq_n_f32(act_max);
                const float32_t inv_count = (count > 0) ? (1.0f / (float32_t)count) : 0.0f;
                for (int32_t c = 0; c < channel_in; c += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)(channel_in - c));
                    float32x4_t v_sum = vdupq_n_f32(0.0f);
                    for (int32_t k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                    {
                        for (int32_t k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                        {
                            const int32_t idx = (k_x + base_idx_x + (k_y + base_idx_y) * input_x) * channel_in + c;
                            v_sum = vaddq(v_sum, vld1q_z(src + idx, p));
                        }
                    }
                    float32x4_t v = vmulq(v_sum, inv_count);
                    v = arm_nn_clamp_mve_f32(v, v_act_min, v_act_max);
                    vst1q_p(dst + (i_y * output_x + i_x) * channel_in + c, v, p);
                }
#else
                for (int32_t c = 0; c < channel_in; ++c)
                {
                    float32_t sum = 0.0f;
                    for (int32_t k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                    {
                        for (int32_t k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                        {
                            const int32_t idx = (k_x + base_idx_x + (k_y + base_idx_y) * input_x) * channel_in + c;
                            sum += src[idx];
                        }
                    }
                    float32_t v = (count > 0) ? (sum / (float32_t)count) : 0.0f;
                    dst[(i_y * output_x + i_x) * channel_in + c] = CLAMP(v, act_max, act_min);
                }
#endif
            }
        }

        src += batch_size;
        dst += output_x * output_y * channel_in;
        batch_cnt--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_avg_pool_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_pool_params_f32 *pool_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *src,
                                          const cmsis_nn_dims *filter_dims,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *dst)
{
    return arm_avg_pool_f32(ctx, pool_params, input_dims, src, filter_dims, output_dims, dst);
}

/**
 * @} end of Pooling group
 */
