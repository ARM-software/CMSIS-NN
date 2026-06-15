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
 * Title:        arm_max_pool_f32.c
 * Description:  Max pooling function implementations (float32)
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
 * Max pooling function for f32
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_max_pool_f32(const cmsis_nn_context *ctx,
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
        bool use_specialized = (input_y == 1 && output_y == 1 && kernel_y == 1 && stride_y == 1 && pad_y == 0 &&
                                kernel_x == 2 && stride_x == 2 && pad_x == 0);
#ifdef NN_DISABLE_SPECIALIZATION
        use_specialized = false;
#endif
        if (use_specialized)
        {
            if (act_min <= ARM_NN_F32_FINITE_LOWEST && act_max >= ARM_NN_F32_FINITE_MAX)
            {
                arm_nn_maxpool1d_k2s2_nhwc_noclip_f32(src, channel_in, input_x, dst, output_x);
            }
            else
            {
                arm_nn_maxpool1d_k2s2_nhwc_f32(src, channel_in, input_x, dst, output_x, act_min, act_max);
            }
            src += batch_size;
            dst += output_x * output_y * channel_in;
            batch_cnt--;
            continue;
        }

        use_specialized =
            (input_y == 1 && output_y == 1 && kernel_y == 1 && stride_y == 1 && pad_y == 0 && kernel_x == 3 &&
             stride_x == 3 && pad_x == 0 && act_min <= ARM_NN_F32_FINITE_LOWEST && act_max >= ARM_NN_F32_FINITE_MAX);
#ifdef NN_DISABLE_SPECIALIZATION
        use_specialized = false;
#endif
        if (use_specialized)
        {
            arm_nn_maxpool1d_k3s3_nhwc_f32(src, channel_in, input_x, dst, output_x);
            src += batch_size;
            dst += output_x * output_y * channel_in;
            batch_cnt--;
            continue;
        }

        for (int32_t i_y = 0, base_idx_y = -pad_y; i_y < output_y; base_idx_y += stride_y, i_y++)
        {
            for (int32_t i_x = 0, base_idx_x = -pad_x; i_x < output_x; base_idx_x += stride_x, i_x++)
            {
                const int32_t ker_y_start = MAX(0, -base_idx_y);
                const int32_t ker_x_start = MAX(0, -base_idx_x);
                const int32_t kernel_y_end = MIN(kernel_y, input_y - base_idx_y);
                const int32_t kernel_x_end = MIN(kernel_x, input_x - base_idx_x);

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                const float32x4_t v_act_min = vdupq_n_f32(act_min);
                const float32x4_t v_act_max = vdupq_n_f32(act_max);
                for (int32_t c = 0; c < channel_in; c += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)(channel_in - c));
                    float32x4_t v_max = vdupq_n_f32(ARM_NN_F32_FINITE_LOWEST);
                    for (int32_t k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                    {
                        for (int32_t k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                        {
                            const int32_t idx = (k_x + base_idx_x + (k_y + base_idx_y) * input_x) * channel_in + c;
                            float32x4_t v = vld1q_z(src + idx, p);
                            v = vpselq(v, v_max, p);
                            v_max = vmaxnmq(v_max, v);
                        }
                    }
                    v_max = arm_nn_clamp_mve_f32(v_max, v_act_min, v_act_max);
                    vst1q_p(dst + (i_y * output_x + i_x) * channel_in + c, v_max, p);
                }
#else
                for (int32_t c = 0; c < channel_in; ++c)
                {
                    float32_t max_val = ARM_NN_F32_FINITE_LOWEST;
                    for (int32_t k_y = ker_y_start; k_y < kernel_y_end; k_y++)
                    {
                        for (int32_t k_x = ker_x_start; k_x < kernel_x_end; k_x++)
                        {
                            const int32_t idx = (k_x + base_idx_x + (k_y + base_idx_y) * input_x) * channel_in + c;
                            max_val = (src[idx] > max_val) ? src[idx] : max_val;
                        }
                    }
                    dst[(i_y * output_x + i_x) * channel_in + c] = CLAMP(max_val, act_max, act_min);
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

/**
 * @} end of Pooling group
 */

arm_cmsis_nn_status arm_max_pool_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_pool_params_f32 *pool_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *src,
                                          const cmsis_nn_dims *filter_dims,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *dst)
{
    return arm_max_pool_f32(ctx, pool_params, input_dims, src, filter_dims, output_dims, dst);
}
