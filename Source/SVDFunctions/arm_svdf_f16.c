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
 * Title:        arm_svdf_f16.c
 * Description:  SVDF layer (float16)
 *
 * $Date:        26 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup SVDF
 * @{
 */

__STATIC_INLINE float16_t clamp_and_activate_f16(float16_t x, const cmsis_nn_activation_f16 *act)
{
    return arm_nn_clamp_scalar_f16(x, act->min, act->max);
}

__STATIC_INLINE float16_t arm_nn_svdf_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t count)
{
    _Float16 sum = (_Float16)0.0f;

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    float16x8_t v_acc = vdupq_n_f16((float16_t)0.0f);
    for (int32_t i = 0; i < count; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(count - i));
        v_acc = vfmaq(v_acc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p));
    }
    sum = arm_nn_vec_reduce_add_f16(v_acc);
#else
    for (int32_t i = 0; i < count; ++i)
    {
        sum += (_Float16)lhs[i] * (_Float16)rhs[i];
    }
#endif

    return (float16_t)sum;
}

arm_cmsis_nn_status arm_svdf_f16(const cmsis_nn_context *ctx,
                                 const cmsis_nn_context *input_ctx,
                                 const cmsis_nn_context *output_ctx,
                                 const cmsis_nn_svdf_params_f16 *svdf_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float16_t *input_data,
                                 const cmsis_nn_dims *state_dims,
                                 float16_t *state_data,
                                 const cmsis_nn_dims *weights_feature_dims,
                                 const float16_t *weights_feature_data,
                                 const cmsis_nn_dims *weights_time_dims,
                                 const float16_t *weights_time_data,
                                 const cmsis_nn_dims *bias_dims,
                                 const float16_t *bias_data,
                                 const cmsis_nn_dims *output_dims,
                                 float16_t *output_data)
{
    (void)ctx;
    (void)state_dims;
    (void)bias_dims;
    (void)output_dims;

    if (!svdf_params || !input_dims || !weights_feature_dims || !weights_time_dims || !input_data || !state_data ||
        !weights_feature_data || !weights_time_data || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = svdf_params->rank;
    const int32_t input_batches = input_dims->n;
    const int32_t input_height = input_dims->h;
    const int32_t feature_batches = weights_feature_dims->n;
    const int32_t time_batches = weights_time_dims->h;
    const int32_t unit_count = feature_batches / rank;

    float16_t *buffer_a = (float16_t *)(input_ctx ? input_ctx->buf : NULL);
    float16_t *buffer_b = (float16_t *)(output_ctx ? output_ctx->buf : NULL);
    if (!buffer_a || !buffer_b)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float16_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            float16_t *state_f = state_b + i_feat * time_batches;
            if (time_batches > 1)
            {
                for (int32_t t = 0; t < time_batches - 1; ++t)
                {
                    state_f[t] = state_f[t + 1];
                }
            }
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const float16_t *input = input_data + i_batch * input_height;
        float16_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            const float16_t *w_feat = weights_feature_data + i_feat * input_height;
            float16_t acc_h = arm_nn_svdf_dot_f16(input, w_feat, input_height);
            acc_h = clamp_and_activate_f16(acc_h, &svdf_params->input_activation);
            state_b[i_feat * time_batches + (time_batches - 1)] = acc_h;
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float16_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        float16_t *out_a = buffer_a + i_batch * feature_batches;
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            const float16_t *w_time = weights_time_data + i_feat * time_batches;
            const float16_t *s_time = state_b + i_feat * time_batches;
            out_a[i_feat] = arm_nn_svdf_dot_f16(w_time, s_time, time_batches);
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float16_t *out_b = buffer_b + i_batch * unit_count;
        float16_t *ptr_a = buffer_a + i_batch * feature_batches;
        if (bias_data)
        {
            if (unit_count == feature_batches)
            {
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                int32_t j = 0;
                for (; j < feature_batches; j += 8)
                {
                    const mve_pred16_t p = vctp16q((uint32_t)(feature_batches - j));
                    const float16x8_t v_a = vld1q_z(ptr_a + j, p);
                    const float16x8_t v_b = vld1q_z(bias_data + j, p);
                    vst1q_p(out_b + j, vaddq(v_a, v_b), p);
                }
#else
                for (int32_t j = 0; j < feature_batches; j++)
                {
                    out_b[j] = (float16_t)((float32_t)ptr_a[j] + (float32_t)bias_data[j]);
                }
#endif
            }
            else
            {
                for (int32_t i = 0; i < unit_count; i++)
                {
                    float32_t sum = (float32_t)bias_data[i];
                    for (int32_t j = 0; j < rank; j++)
                    {
                        sum += (float32_t)(*ptr_a++);
                    }
                    out_b[i] = (float16_t)sum;
                }
            }
        }
        else
        {
            for (int32_t i = 0; i < unit_count; i++)
            {
                float32_t sum = 0.0f;
                for (int32_t j = 0; j < rank; j++)
                {
                    sum += (float32_t)(*ptr_a++);
                }
                out_b[i] = (float16_t)sum;
            }
        }
    }

    const int32_t output_count = input_batches * unit_count;
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t v_act_min = vdupq_n_f16(svdf_params->output_activation.min);
    const float16x8_t v_act_max = vdupq_n_f16(svdf_params->output_activation.max);
    for (int32_t i = 0; i < output_count; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(output_count - i));
        float16x8_t v = vld1q_z(buffer_b + i, p);
        v = arm_nn_clamp_mve_f16(v, v_act_min, v_act_max);
        vst1q_p(output_data + i, v, p);
    }
#else
    for (int32_t i = 0; i < output_count; i++)
    {
        output_data[i] = clamp_and_activate_f16(buffer_b[i], &svdf_params->output_activation);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of SVDF group
 */
