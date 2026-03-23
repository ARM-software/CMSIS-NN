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
 * Title:        arm_conv_opt_f16.h
 * Description:  Float16 convolution specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_OPT_F16_H
#define ARM_CONV_OPT_F16_H

/* Internal specialization helpers (included from arm_convolve_f16.c). */

#include "Internal/arm_conv_opt_common.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION
typedef bool (*arm_conv_match_f16)(const cmsis_nn_context *ctx,
                                   const cmsis_nn_conv_params_f16 *params,
                                   const cmsis_nn_dims *input_dims,
                                   const float16_t *input_data,
                                   const cmsis_nn_dims *filter_dims,
                                   const float16_t *filter_data,
                                   const cmsis_nn_dims *bias_dims,
                                   const float16_t *bias_data,
                                   const cmsis_nn_dims *output_dims,
                                   float16_t *output_data);

typedef arm_cmsis_nn_status (*arm_conv_call_f16)(const cmsis_nn_context *ctx,
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
    arm_conv_match_f16 match;
    arm_conv_call_f16 call;
} arm_conv_spec_f16;

static bool arm_conv1d_spec_k5_nhwc_f16_match(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f16 *params,
                                              const cmsis_nn_dims *input_dims,
                                              const float16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float16_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float16_t *output_data)
{
    (void)ctx;
    (void)input_data;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_data;

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t output_h = output_dims->h;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = params->stride.h;
    const int32_t stride_w = params->stride.w;
    const int32_t pad_h = params->padding.h;
    const int32_t pad_w = params->padding.w;
    const int32_t dil_h = params->dilation.h;
    const int32_t dil_w = params->dilation.w;

    return (batch == 1 && input_h == 1 && output_h == 1 && kernel_h == 1 && kernel_w == 5 && stride_h == 1 &&
            stride_w == 1 && pad_h == 0 && pad_w == 0 && dil_h == 1 && dil_w == 1);
}

static arm_cmsis_nn_status arm_conv1d_spec_k5_nhwc_f16_call(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_conv_params_f16 *params,
                                                            const cmsis_nn_dims *input_dims,
                                                            const float16_t *input_data,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const float16_t *filter_data,
                                                            const cmsis_nn_dims *bias_dims,
                                                            const float16_t *bias_data,
                                                            const cmsis_nn_dims *output_dims,
                                                            float16_t *output_data)
{
    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t output_w = output_dims->w;

    if (params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        arm_nn_conv1d_k5_packed_f16(
            input_data, input_c, input_w, filter_data, bias_data, output_data, output_c, output_w);
    }
    else
    {
        arm_nn_conv1d_k5_nhwc_f16(
            input_data, input_c, input_w, filter_data, bias_data, output_data, output_c, output_w);
    }

    const int32_t out_count = output_c * output_w * output_dims->h;
    arm_nn_vector_clamp_f16(output_data, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

static bool arm_conv1d_spec_k3_nhwc_f16_match(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f16 *params,
                                              const cmsis_nn_dims *input_dims,
                                              const float16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float16_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float16_t *output_data)
{
    (void)ctx;
    (void)input_data;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_data;

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t output_h = output_dims->h;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = params->stride.h;
    const int32_t stride_w = params->stride.w;
    const int32_t pad_h = params->padding.h;
    const int32_t pad_w = params->padding.w;
    const int32_t dil_h = params->dilation.h;
    const int32_t dil_w = params->dilation.w;

    return (batch == 1 && input_h == 1 && output_h == 1 && kernel_h == 1 && kernel_w == 3 && stride_h == 1 &&
            stride_w == 1 && pad_h == 0 && pad_w == 0 && dil_h == 1 && dil_w == 1);
}

static arm_cmsis_nn_status arm_conv1d_spec_k3_nhwc_f16_call(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_conv_params_f16 *params,
                                                            const cmsis_nn_dims *input_dims,
                                                            const float16_t *input_data,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const float16_t *filter_data,
                                                            const cmsis_nn_dims *bias_dims,
                                                            const float16_t *bias_data,
                                                            const cmsis_nn_dims *output_dims,
                                                            float16_t *output_data)
{
    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t output_w = output_dims->w;

    if (params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        arm_nn_conv1d_k3_packed_f16(
            input_data, input_c, input_w, filter_data, bias_data, output_data, output_c, output_w);
    }
    else
    {
        arm_nn_conv1d_k3_nhwc_f16(
            input_data, input_c, input_w, filter_data, bias_data, output_data, output_c, output_w);
    }

    const int32_t out_count = output_c * output_w * output_dims->h;
    arm_nn_vector_clamp_f16(output_data, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

static const arm_conv_spec_f16 arm_conv_spec_nhwc_f16[] = {
    ARM_CONV_SPEC_ENTRY(arm_conv1d_spec_k5_nhwc_f16_match, arm_conv1d_spec_k5_nhwc_f16_call),
    ARM_CONV_SPEC_ENTRY(arm_conv1d_spec_k3_nhwc_f16_match, arm_conv1d_spec_k3_nhwc_f16_call),
};

__STATIC_INLINE bool arm_conv_spec_nhwc_f16_matches_any(const cmsis_nn_context *ctx,
                                                        const cmsis_nn_conv_params_f16 *params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const cmsis_nn_dims *output_dims)
{
    for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(arm_conv_spec_nhwc_f16); ++i)
    {
        if (arm_conv_spec_nhwc_f16[i].match(
                ctx, params, input_dims, NULL, filter_dims, NULL, NULL, NULL, output_dims, NULL))
        {
            return true;
        }
    }

    return false;
}
#endif

#endif /* ARM_CONV_OPT_F16_H */
