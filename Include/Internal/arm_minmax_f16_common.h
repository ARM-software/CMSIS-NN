/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_minmax_f16_common.h
 * Description:  Shared float16 min/max helper templates
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_MINMAX_F16_COMMON_H
#define ARM_MINMAX_F16_COMMON_H

#include "arm_nn_compiler.h"
#include "arm_nn_types_flt.h"

__STATIC_INLINE int32_t arm_check_broadcast_required_f16(const cmsis_nn_dims *shape_1, const cmsis_nn_dims *shape_2)
{
    if ((shape_1->n != shape_2->n) || (shape_1->h != shape_2->h) || (shape_1->w != shape_2->w) ||
        (shape_1->c != shape_2->c))
    {
        return 1;
    }

    return 0;
}

__STATIC_INLINE float16_t arm_minmax_select_f16(float16_t a, float16_t b, int32_t select_max)
{
    return ((_Float16)a >= (_Float16)b) ? (select_max ? (_Float16)a : (_Float16)b)
                                        : (select_max ? (_Float16)b : (_Float16)a);
}

__STATIC_INLINE arm_cmsis_nn_status arm_minmax_no_broadcast_f16(const float16_t *input_1,
                                                                const float16_t *input_2,
                                                                float16_t *output,
                                                                int32_t flat_size,
                                                                int32_t select_max)
{
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in1 = vld1q_z(input_1 + i, p);
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in1 = vld1q_z(input_1 + i, p);
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
#else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f16(input_1[i], input_2[i], select_max);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_INLINE arm_cmsis_nn_status arm_minmax_scalar_f16(const float16_t *input_1,
                                                          const float16_t *input_2,
                                                          float16_t *output,
                                                          int32_t flat_size,
                                                          int32_t select_max)
{
    const float16_t in1 = *input_1;
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t v_in1 = vdupq_n_f16(in1);
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
#else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f16(in1, input_2[i], select_max);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_INLINE arm_cmsis_nn_status arm_minmax_f16_impl(const cmsis_nn_context *ctx,
                                                        const float16_t *input_1_data,
                                                        const cmsis_nn_dims *input_1_dims,
                                                        const float16_t *input_2_data,
                                                        const cmsis_nn_dims *input_2_dims,
                                                        float16_t *output_data,
                                                        const cmsis_nn_dims *output_dims,
                                                        int32_t select_max)
{
    (void)ctx;
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t output_batch = output_dims->n;
    const int32_t output_height = output_dims->h;
    const int32_t output_width = output_dims->w;

    const int32_t input_1_batch = input_1_dims->n;
    const int32_t input_1_height = input_1_dims->h;
    const int32_t input_1_width = input_1_dims->w;
    const int32_t input_1_channels = input_1_dims->c;

    const int32_t input_2_batch = input_2_dims->n;
    const int32_t input_2_height = input_2_dims->h;
    const int32_t input_2_width = input_2_dims->w;
    const int32_t input_2_channels = input_2_dims->c;

    int32_t flat_size_1 = input_1_batch * input_1_height * input_1_width * input_1_channels;
    int32_t flat_size_2 = input_2_batch * input_2_height * input_2_width * input_2_channels;

    if (arm_check_broadcast_required_f16(input_1_dims, input_2_dims))
    {
        if (flat_size_1 == 1)
        {
            arm_minmax_scalar_f16(input_1_data, input_2_data, output_data, flat_size_2, select_max);
        }
        else if (flat_size_2 == 1)
        {
            arm_minmax_scalar_f16(input_2_data, input_1_data, output_data, flat_size_1, select_max);
        }
        else
        {
            const int32_t width_1_diff = input_1_width >= input_2_width ? 0 : input_1_channels;
            const int32_t width_2_diff = input_2_width >= input_1_width ? 0 : input_2_channels;

            const int32_t height_1_diff =
                input_1_height >= input_2_height ? width_1_diff : -input_1_width * (input_1_channels - width_1_diff);
            const int32_t height_2_diff =
                input_2_height >= input_1_height ? width_2_diff : -input_2_width * (input_2_channels - width_2_diff);

            const int32_t batch_1_diff =
                input_1_batch >= input_2_batch ? input_1_channels * input_1_width * input_1_height : 0;
            const int32_t batch_2_diff =
                input_2_batch >= input_1_batch ? input_2_channels * input_2_width * input_2_height : 0;

            for (int32_t i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
            {
                const float16_t *input_1_ptr = input_1_data;
                const float16_t *input_2_ptr = input_2_data;
                flat_size_1 = input_1_height * input_1_width * input_1_channels;
                flat_size_2 = input_2_height * input_2_width * input_2_channels;
                if (input_1_height == input_2_height && input_1_width == input_2_width &&
                    input_1_channels == input_2_channels)
                {
                    arm_minmax_no_broadcast_f16(input_1_ptr, input_2_ptr, output_data, flat_size_1, select_max);
                    output_data += flat_size_1;
                }
                else if (flat_size_1 == 1)
                {
                    arm_minmax_scalar_f16(input_1_ptr, input_2_ptr, output_data, flat_size_2, select_max);
                    output_data += flat_size_2;
                }
                else if (flat_size_2 == 1)
                {
                    arm_minmax_scalar_f16(input_2_ptr, input_1_ptr, output_data, flat_size_1, select_max);
                    output_data += flat_size_1;
                }
                else
                {
                    flat_size_1 = input_1_width * input_1_channels;
                    flat_size_2 = input_2_width * input_2_channels;
                    for (int32_t i_out_height = 0; i_out_height < output_height; i_out_height++)
                    {
                        if (input_1_width == input_2_width && input_1_channels == input_2_channels)
                        {
                            arm_minmax_no_broadcast_f16(input_1_ptr, input_2_ptr, output_data, flat_size_1, select_max);
                            output_data += flat_size_1;
                            input_1_ptr += flat_size_1;
                            input_2_ptr += flat_size_1;
                        }
                        else if (flat_size_1 == 1)
                        {
                            arm_minmax_scalar_f16(input_1_ptr, input_2_ptr, output_data, flat_size_2, select_max);
                            output_data += flat_size_2;
                            ++input_1_ptr;
                            input_2_ptr += flat_size_2;
                        }
                        else if (flat_size_2 == 1)
                        {
                            arm_minmax_scalar_f16(input_2_ptr, input_1_ptr, output_data, flat_size_1, select_max);
                            output_data += flat_size_1;
                            ++input_2_ptr;
                            input_1_ptr += flat_size_1;
                        }
                        else
                        {
                            for (int32_t i_out_width = 0; i_out_width < output_width; i_out_width++)
                            {
                                if (input_1_channels == input_2_channels)
                                {
                                    arm_minmax_no_broadcast_f16(
                                        input_1_ptr, input_2_ptr, output_data, input_1_channels, select_max);
                                    output_data += input_1_channels;
                                    input_1_ptr += input_1_channels;
                                    input_2_ptr += input_1_channels;
                                }
                                else if (input_1_channels == 1)
                                {
                                    arm_minmax_scalar_f16(
                                        input_1_ptr, input_2_ptr, output_data, input_2_channels, select_max);
                                    output_data += input_2_channels;
                                    input_1_ptr++;
                                    input_2_ptr += input_2_channels;
                                }
                                else if (input_2_channels == 1)
                                {
                                    arm_minmax_scalar_f16(
                                        input_2_ptr, input_1_ptr, output_data, input_1_channels, select_max);
                                    output_data += input_1_channels;
                                    input_1_ptr += input_1_channels;
                                    input_2_ptr++;
                                }
                                input_1_ptr -= width_1_diff;
                                input_2_ptr -= width_2_diff;
                            }
                        }
                        input_1_ptr += height_1_diff;
                        input_2_ptr += height_2_diff;
                    }
                }
                input_1_data += batch_1_diff;
                input_2_data += batch_2_diff;
            }
        }
    }
    else
    {
        arm_minmax_no_broadcast_f16(input_1_data, input_2_data, output_data, flat_size_1, select_max);
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_MINMAX_F16_COMMON_H */
