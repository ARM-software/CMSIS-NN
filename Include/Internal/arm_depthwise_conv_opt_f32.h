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
 * Title:        arm_depthwise_conv_opt_f32.h
 * Description:  Float32 depthwise-convolution specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_DEPTHWISE_CONV_OPT_F32_H
#define ARM_DEPTHWISE_CONV_OPT_F32_H

/* Internal specialization helpers (included from arm_depthwise_conv_f32.c). */

#include "Internal/arm_depthwise_conv_opt_common.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION
typedef bool (*arm_dw_match_f32)(const cmsis_nn_context *ctx,
                                 const cmsis_nn_dw_conv_params_f32 *params,
                                 const cmsis_nn_dims *input_dims,
                                 const float32_t *input,
                                 const cmsis_nn_dims *filter_dims,
                                 const float32_t *kernel,
                                 const cmsis_nn_dims *bias_dims,
                                 const float32_t *bias,
                                 const cmsis_nn_dims *output_dims,
                                 float32_t *output,
                                 arm_nn_dw_kernel_layout_f32 kernel_layout);

typedef arm_cmsis_nn_status (*arm_dw_call_f32)(const cmsis_nn_context *ctx,
                                               const cmsis_nn_dw_conv_params_f32 *params,
                                               const cmsis_nn_dims *input_dims,
                                               const float32_t *input,
                                               const cmsis_nn_dims *filter_dims,
                                               const float32_t *kernel,
                                               const cmsis_nn_dims *bias_dims,
                                               const float32_t *bias,
                                               const cmsis_nn_dims *output_dims,
                                               float32_t *output,
                                               arm_nn_dw_kernel_layout_f32 kernel_layout);

typedef struct
{
    arm_dw_match_f32 match;
    arm_dw_call_f32 call;
} arm_dw_spec_f32;

static bool arm_dw_spec_k3_1d_nhwc_f32_match(const cmsis_nn_context *ctx,
                                             const cmsis_nn_dw_conv_params_f32 *params,
                                             const cmsis_nn_dims *input_dims,
                                             const float32_t *input,
                                             const cmsis_nn_dims *filter_dims,
                                             const float32_t *kernel,
                                             const cmsis_nn_dims *bias_dims,
                                             const float32_t *bias,
                                             const cmsis_nn_dims *output_dims,
                                             float32_t *output,
                                             arm_nn_dw_kernel_layout_f32 kernel_layout)
{
    (void)ctx;
    (void)input;
    (void)kernel;
    (void)bias_dims;
    (void)bias;
    (void)output;

    const int32_t ch_mult = params->ch_mult;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t batch = input_dims->n;
    const int32_t output_batch = output_dims->n;
    const int32_t input_y = input_dims->h;
    const int32_t output_y = output_dims->h;

    return (kernel_layout == ARM_NN_DW_KERNEL_KC && batch == 1 && output_batch == 1 && ch_mult == 1 && kernel_x == 3 &&
            kernel_y == 1 && input_y == 1 && output_y == 1 && params->dilation.h == 1 && params->dilation.w == 1 &&
            params->stride.h == 1 && params->stride.w == 1 && params->padding.h == 0 && params->padding.w == 0);
}

static arm_cmsis_nn_status arm_dw_spec_k3_1d_nhwc_f32_call(const cmsis_nn_context *ctx,
                                                           const cmsis_nn_dw_conv_params_f32 *params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float32_t *input,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const float32_t *kernel,
                                                           const cmsis_nn_dims *bias_dims,
                                                           const float32_t *bias,
                                                           const cmsis_nn_dims *output_dims,
                                                           float32_t *output,
                                                           arm_nn_dw_kernel_layout_f32 kernel_layout)
{
    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

    if (kernel_layout != ARM_NN_DW_KERNEL_KC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    arm_nn_depthwise_conv1d_k3_nhwc_f32(input, input_dims->c, input_dims->w, kernel, bias, output, output_dims->w);

    const int32_t out_count = output_dims->n * output_dims->c * output_dims->h * output_dims->w;
    arm_nn_vector_clamp_f32(output, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

static bool arm_dw_spec_3x3_nhwc_f32_match(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f32 *params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output,
                                           arm_nn_dw_kernel_layout_f32 kernel_layout)
{
    (void)ctx;
    (void)input;
    (void)kernel;
    (void)bias_dims;
    (void)bias;
    (void)output;

    const int32_t batch = input_dims->n;
    const int32_t output_batch = output_dims->n;

    return (kernel_layout == ARM_NN_DW_KERNEL_KC && batch > 0 && batch == output_batch && params->ch_mult == 1 &&
            filter_dims->w == 3 && filter_dims->h == 3 && params->dilation.h == 1 && params->dilation.w == 1);
}

static arm_cmsis_nn_status arm_dw_spec_3x3_nhwc_f32_call(const cmsis_nn_context *ctx,
                                                         const cmsis_nn_dw_conv_params_f32 *params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const float32_t *input,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const float32_t *kernel,
                                                         const cmsis_nn_dims *bias_dims,
                                                         const float32_t *bias,
                                                         const cmsis_nn_dims *output_dims,
                                                         float32_t *output,
                                                         arm_nn_dw_kernel_layout_f32 kernel_layout)
{
    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

    if (kernel_layout != ARM_NN_DW_KERNEL_KC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    arm_nn_depthwise_conv3x3_nhwc_f32(input,
                                      input_dims->n,
                                      input_dims->c,
                                      input_dims->h,
                                      input_dims->w,
                                      kernel,
                                      bias,
                                      output,
                                      params->stride.w,
                                      params->stride.h,
                                      params->padding.w,
                                      params->padding.h,
                                      output_dims->h,
                                      output_dims->w,
                                      params->activation.min,
                                      params->activation.max);
    return ARM_CMSIS_NN_SUCCESS;
}

static const arm_dw_spec_f32 arm_dw_spec_nhwc_f32[] = {
    ARM_DW_SPEC_ENTRY(arm_dw_spec_k3_1d_nhwc_f32_match, arm_dw_spec_k3_1d_nhwc_f32_call),
    ARM_DW_SPEC_ENTRY(arm_dw_spec_3x3_nhwc_f32_match, arm_dw_spec_3x3_nhwc_f32_call),
};
#endif

#endif /* ARM_DEPTHWISE_CONV_OPT_F32_H */
