/*
 * SPDX-FileCopyrightText: Copyright 2010-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "../TestData/basic/test_data.h"
#include "../TestData/conv_2/test_data.h"
#include "../TestData/conv_2x2_dilation/test_data.h"
#include "../TestData/conv_2x2_dilation_5x5_input/test_data.h"
#include "../TestData/conv_2x3_dilation/test_data.h"
#include "../TestData/conv_3/test_data.h"
#include "../TestData/conv_3x2_dilation/test_data.h"
#include "../TestData/conv_3x3_dilation_5x5_input/test_data.h"
#include "../TestData/conv_4/test_data.h"
#include "../TestData/conv_5/test_data.h"
#include "../TestData/conv_dilation_golden/test_data.h"
#include "../TestData/conv_out_activation/test_data.h"
#include "../TestData/stride2pad1/test_data.h"
#include "../Utils/validate.h"
#include "../Utils/validate_aot.h"

void basic_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_biases;
    const int8_t *kernel_data = basic_weights;
    const int8_t *input_data = basic_input;
    const int8_t *output_ref = basic_output_ref;
    const int32_t output_ref_size = BASIC_DST_SIZE;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    filter_dims.c = BASIC_IN_CH;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    conv_params.padding.w = BASIC_PAD_X;
    conv_params.padding.h = BASIC_PAD_Y;
    conv_params.stride.w = BASIC_STRIDE_X;
    conv_params.stride.h = BASIC_STRIDE_Y;
    conv_params.dilation.w = BASIC_DILATION_X;
    conv_params.dilation.h = BASIC_DILATION_Y;

    conv_params.input_offset = BASIC_INPUT_OFFSET;
    conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void stride2pad1_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = stride2pad1_biases;
    const int8_t *kernel_data = stride2pad1_weights;
    const int8_t *input_data = stride2pad1_input;
    const int8_t *output_ref = stride2pad1_output_ref;
    const int32_t output_ref_size = STRIDE2PAD1_DST_SIZE;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    filter_dims.c = STRIDE2PAD1_IN_CH;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    conv_params.padding.w = STRIDE2PAD1_PAD_X;
    conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;
    conv_params.dilation.w = STRIDE2PAD1_DILATION_X;
    conv_params.dilation.h = STRIDE2PAD1_DILATION_Y;

    conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_2_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2_biases;
    const int8_t *kernel_data = conv_2_weights;
    const int8_t *input_data = conv_2_input;
    const int8_t *output_ref = conv_2_output_ref;
    const int32_t output_ref_size = CONV_2_DST_SIZE;

    input_dims.n = CONV_2_INPUT_BATCHES;
    input_dims.w = CONV_2_INPUT_W;
    input_dims.h = CONV_2_INPUT_H;
    input_dims.c = CONV_2_IN_CH;
    filter_dims.w = CONV_2_FILTER_X;
    filter_dims.h = CONV_2_FILTER_Y;
    filter_dims.c = CONV_2_IN_CH;
    output_dims.w = CONV_2_OUTPUT_W;
    output_dims.h = CONV_2_OUTPUT_H;
    output_dims.c = CONV_2_OUT_CH;

    conv_params.padding.w = CONV_2_PAD_X;
    conv_params.padding.h = CONV_2_PAD_Y;
    conv_params.stride.w = CONV_2_STRIDE_X;
    conv_params.stride.h = CONV_2_STRIDE_Y;
    conv_params.dilation.w = CONV_2_DILATION_X;
    conv_params.dilation.h = CONV_2_DILATION_Y;

    conv_params.input_offset = CONV_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2_output_mult;
    quant_params.shift = (int32_t *)conv_2_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_2_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_3_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3_biases;
    const int8_t *kernel_data = conv_3_weights;
    const int8_t *input_data = conv_3_input;
    const int8_t *output_ref = conv_3_output_ref;
    const int32_t output_ref_size = CONV_3_DST_SIZE;

    input_dims.n = CONV_3_INPUT_BATCHES;
    input_dims.w = CONV_3_INPUT_W;
    input_dims.h = CONV_3_INPUT_H;
    input_dims.c = CONV_3_IN_CH;
    filter_dims.w = CONV_3_FILTER_X;
    filter_dims.h = CONV_3_FILTER_Y;
    filter_dims.c = CONV_3_IN_CH;
    output_dims.w = CONV_3_OUTPUT_W;
    output_dims.h = CONV_3_OUTPUT_H;
    output_dims.c = CONV_3_OUT_CH;

    conv_params.padding.w = CONV_3_PAD_X;
    conv_params.padding.h = CONV_3_PAD_Y;
    conv_params.stride.w = CONV_3_STRIDE_X;
    conv_params.stride.h = CONV_3_STRIDE_Y;
    conv_params.dilation.w = CONV_3_DILATION_X;
    conv_params.dilation.h = CONV_3_DILATION_Y;

    conv_params.input_offset = CONV_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3_output_mult;
    quant_params.shift = (int32_t *)conv_3_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_3_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_4_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_4_biases;
    const int8_t *kernel_data = conv_4_weights;
    const int8_t *input_data = conv_4_input;
    const int8_t *output_ref = conv_4_output_ref;
    const int32_t output_ref_size = CONV_4_DST_SIZE;

    input_dims.n = CONV_4_INPUT_BATCHES;
    input_dims.w = CONV_4_INPUT_W;
    input_dims.h = CONV_4_INPUT_H;
    input_dims.c = CONV_4_IN_CH;
    filter_dims.w = CONV_4_FILTER_X;
    filter_dims.h = CONV_4_FILTER_Y;
    filter_dims.c = CONV_4_IN_CH;
    output_dims.w = CONV_4_OUTPUT_W;
    output_dims.h = CONV_4_OUTPUT_H;
    output_dims.c = CONV_4_OUT_CH;

    conv_params.padding.w = CONV_4_PAD_X;
    conv_params.padding.h = CONV_4_PAD_Y;
    conv_params.stride.w = CONV_4_STRIDE_X;
    conv_params.stride.h = CONV_4_STRIDE_Y;
    conv_params.dilation.w = CONV_4_DILATION_X;
    conv_params.dilation.h = CONV_4_DILATION_Y;

    conv_params.input_offset = CONV_4_INPUT_OFFSET;
    conv_params.output_offset = CONV_4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_4_output_mult;
    quant_params.shift = (int32_t *)conv_4_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_out_activation_arm_convolve_s8(void)
{
    int8_t output[CONV_OUT_ACTIVATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_out_activation_biases;
    const int8_t *kernel_data = conv_out_activation_weights;
    const int8_t *input_data = conv_out_activation_input;
    const int8_t *output_ref = conv_out_activation_output_ref;
    const int32_t output_ref_size = CONV_OUT_ACTIVATION_DST_SIZE;

    input_dims.n = CONV_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = CONV_OUT_ACTIVATION_INPUT_W;
    input_dims.h = CONV_OUT_ACTIVATION_INPUT_H;
    input_dims.c = CONV_OUT_ACTIVATION_IN_CH;
    filter_dims.w = CONV_OUT_ACTIVATION_FILTER_X;
    filter_dims.h = CONV_OUT_ACTIVATION_FILTER_Y;
    filter_dims.c = CONV_OUT_ACTIVATION_IN_CH;
    output_dims.w = CONV_OUT_ACTIVATION_OUTPUT_W;
    output_dims.h = CONV_OUT_ACTIVATION_OUTPUT_H;
    output_dims.c = CONV_OUT_ACTIVATION_OUT_CH;

    conv_params.padding.w = CONV_OUT_ACTIVATION_PAD_X;
    conv_params.padding.h = CONV_OUT_ACTIVATION_PAD_Y;
    conv_params.stride.w = CONV_OUT_ACTIVATION_STRIDE_X;
    conv_params.stride.h = CONV_OUT_ACTIVATION_STRIDE_Y;
    conv_params.dilation.w = CONV_OUT_ACTIVATION_DILATION_X;
    conv_params.dilation.h = CONV_OUT_ACTIVATION_DILATION_Y;

    conv_params.input_offset = CONV_OUT_ACTIVATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_OUT_ACTIVATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_out_activation_output_mult;
    quant_params.shift = (int32_t *)conv_out_activation_output_shift;

    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_wrapper_s8(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_2x2_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_2X2_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int32_t *bias_data = conv_2x2_dilation_biases;
    const int8_t *kernel_data = conv_2x2_dilation_weights;
    const int8_t *input_data = conv_2x2_dilation_input;
    const int8_t *output_ref = conv_2x2_dilation_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_DST_SIZE;

    input_dims.n = CONV_2X2_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_FILTER_Y;
    filter_dims.c = CONV_2X2_DILATION_IN_CH;
    output_dims.w = CONV_2X2_DILATION_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_2x2_dilation_5x5_input_arm_convolve_s8(void)
{
    int8_t output[CONV_2X2_DILATION_5X5_INPUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x2_dilation_5x5_input_biases;
    const int8_t *kernel_data = conv_2x2_dilation_5x5_input_weights;
    const int8_t *input_data = conv_2x2_dilation_5x5_input_input;
    const int8_t *output_ref = conv_2x2_dilation_5x5_input_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_5X5_INPUT_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X2_DILATION_5X5_INPUT_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_5X5_INPUT_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_5X5_INPUT_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_5X5_INPUT_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_5X5_INPUT_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_5X5_INPUT_FILTER_Y;
    filter_dims.c = CONV_2X2_DILATION_5X5_INPUT_IN_CH;
    output_dims.w = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_5X5_INPUT_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_5X5_INPUT_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_5X5_INPUT_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_5X5_INPUT_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_5X5_INPUT_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_5X5_INPUT_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_5X5_INPUT_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_5X5_INPUT_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_5X5_INPUT_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_5X5_INPUT_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_5x5_input_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_5x5_input_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_3x3_dilation_5x5_input_arm_convolve_s8(void)
{
    int8_t output[CONV_3X3_DILATION_5X5_INPUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x3_dilation_5x5_input_biases;
    const int8_t *kernel_data = conv_3x3_dilation_5x5_input_weights;
    const int8_t *input_data = conv_3x3_dilation_5x5_input_input;
    const int8_t *output_ref = conv_3x3_dilation_5x5_input_output_ref;
    const int32_t output_ref_size = CONV_3X3_DILATION_5X5_INPUT_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X3_DILATION_5X5_INPUT_INPUT_BATCHES;
    input_dims.w = CONV_3X3_DILATION_5X5_INPUT_INPUT_W;
    input_dims.h = CONV_3X3_DILATION_5X5_INPUT_INPUT_H;
    input_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    filter_dims.w = CONV_3X3_DILATION_5X5_INPUT_FILTER_X;
    filter_dims.h = CONV_3X3_DILATION_5X5_INPUT_FILTER_Y;
    filter_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    output_dims.w = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_W;
    output_dims.h = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_H;
    output_dims.c = CONV_3X3_DILATION_5X5_INPUT_OUT_CH;

    conv_params.padding.w = CONV_3X3_DILATION_5X5_INPUT_PAD_X;
    conv_params.padding.h = CONV_3X3_DILATION_5X5_INPUT_PAD_Y;
    conv_params.stride.w = CONV_3X3_DILATION_5X5_INPUT_STRIDE_X;
    conv_params.stride.h = CONV_3X3_DILATION_5X5_INPUT_STRIDE_Y;
    conv_params.dilation.w = CONV_3X3_DILATION_5X5_INPUT_DILATION_X;
    conv_params.dilation.h = CONV_3X3_DILATION_5X5_INPUT_DILATION_Y;

    conv_params.input_offset = CONV_3X3_DILATION_5X5_INPUT_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x3_dilation_5x5_input_output_mult;
    quant_params.shift = (int32_t *)conv_3x3_dilation_5x5_input_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_2x3_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_2X3_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x3_dilation_biases;
    const int8_t *kernel_data = conv_2x3_dilation_weights;
    const int8_t *input_data = conv_2x3_dilation_input;
    const int8_t *output_ref = conv_2x3_dilation_output_ref;
    const int32_t output_ref_size = CONV_2X3_DILATION_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X3_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_2X3_DILATION_INPUT_W;
    input_dims.h = CONV_2X3_DILATION_INPUT_H;
    input_dims.c = CONV_2X3_DILATION_IN_CH;
    filter_dims.w = CONV_2X3_DILATION_FILTER_X;
    filter_dims.h = CONV_2X3_DILATION_FILTER_Y;
    filter_dims.c = CONV_2X3_DILATION_IN_CH;
    output_dims.w = CONV_2X3_DILATION_OUTPUT_W;
    output_dims.h = CONV_2X3_DILATION_OUTPUT_H;
    output_dims.c = CONV_2X3_DILATION_OUT_CH;

    conv_params.padding.w = CONV_2X3_DILATION_PAD_X;
    conv_params.padding.h = CONV_2X3_DILATION_PAD_Y;
    conv_params.stride.w = CONV_2X3_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_2X3_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_2X3_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_2X3_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_2X3_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X3_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X3_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X3_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x3_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_2x3_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_3x2_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_3X2_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x2_dilation_biases;
    const int8_t *kernel_data = conv_3x2_dilation_weights;
    const int8_t *input_data = conv_3x2_dilation_input;
    const int8_t *output_ref = conv_3x2_dilation_output_ref;
    const int32_t output_ref_size = CONV_3X2_DILATION_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X2_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_3X2_DILATION_INPUT_W;
    input_dims.h = CONV_3X2_DILATION_INPUT_H;
    input_dims.c = CONV_3X2_DILATION_IN_CH;
    filter_dims.w = CONV_3X2_DILATION_FILTER_X;
    filter_dims.h = CONV_3X2_DILATION_FILTER_Y;
    filter_dims.c = CONV_3X2_DILATION_IN_CH;
    output_dims.w = CONV_3X2_DILATION_OUTPUT_W;
    output_dims.h = CONV_3X2_DILATION_OUTPUT_H;
    output_dims.c = CONV_3X2_DILATION_OUT_CH;

    conv_params.padding.w = CONV_3X2_DILATION_PAD_X;
    conv_params.padding.h = CONV_3X2_DILATION_PAD_Y;
    conv_params.stride.w = CONV_3X2_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_3X2_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_3X2_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_3X2_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_3X2_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X2_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X2_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X2_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x2_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_3x2_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_dilation_golden_arm_convolve_s8(void)
{
    int8_t output[CONV_DILATION_GOLDEN_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_dilation_golden_biases;
    const int8_t *kernel_data = conv_dilation_golden_weights;
    const int8_t *input_data = conv_dilation_golden_input;
    const int8_t *output_ref = conv_dilation_golden_output_ref;
    const int32_t output_ref_size = CONV_DILATION_GOLDEN_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_DILATION_GOLDEN_INPUT_BATCHES;
    input_dims.w = CONV_DILATION_GOLDEN_INPUT_W;
    input_dims.h = CONV_DILATION_GOLDEN_INPUT_H;
    input_dims.c = CONV_DILATION_GOLDEN_IN_CH;
    filter_dims.w = CONV_DILATION_GOLDEN_FILTER_X;
    filter_dims.h = CONV_DILATION_GOLDEN_FILTER_Y;
    filter_dims.c = CONV_DILATION_GOLDEN_IN_CH;
    output_dims.w = CONV_DILATION_GOLDEN_OUTPUT_W;
    output_dims.h = CONV_DILATION_GOLDEN_OUTPUT_H;
    output_dims.c = CONV_DILATION_GOLDEN_OUT_CH;

    conv_params.padding.w = CONV_DILATION_GOLDEN_PAD_X;
    conv_params.padding.h = CONV_DILATION_GOLDEN_PAD_Y;
    conv_params.stride.w = CONV_DILATION_GOLDEN_STRIDE_X;
    conv_params.stride.h = CONV_DILATION_GOLDEN_STRIDE_Y;
    conv_params.dilation.w = CONV_DILATION_GOLDEN_DILATION_X;
    conv_params.dilation.h = CONV_DILATION_GOLDEN_DILATION_Y;

    conv_params.input_offset = CONV_DILATION_GOLDEN_INPUT_OFFSET;
    conv_params.output_offset = CONV_DILATION_GOLDEN_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_DILATION_GOLDEN_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_DILATION_GOLDEN_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_dilation_golden_output_mult;
    quant_params.shift = (int32_t *)conv_dilation_golden_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void conv_5_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_5_biases;
    const int8_t *kernel_data = conv_5_weights;
    const int8_t *input_data = conv_5_input;
    const int8_t *output_ref = conv_5_output_ref;
    const int32_t output_ref_size = CONV_5_DST_SIZE;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_5_output_mult;
    quant_params.shift = (int32_t *)conv_5_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_5_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    validate_aot_convolve_s8(&conv_params,
                             &quant_params,
                             &input_dims,
                             input_data,
                             &filter_dims,
                             kernel_data,
                             &bias_dims,
                             bias_data,
                             &output_dims,
                             output_ref);
}

void buffer_size_arm_convolve_s8(void)
{
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_convolve_s8(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_convolve_s8(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

void packed_convolve_s8_arguments(void)
{
#if ARM_NN_ENABLE_AOT
    cmsis_nn_conv_params params = {0};
    cmsis_nn_dims shape = {1, 1, 1, 1};
    int32_t bias = 0;
    const int32_t formats[][2] = {{0, 1}, {1, 0}, {2, 0}, {1, 2}, {-1, -1}, {1, 1}};
    for (unsigned i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
    {
        params.filter_format = formats[i][0];
        params.bias_format = formats[i][1];
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_s8(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_1x1_s8(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_convolve_1x1_s8_fast(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_convolve_wrapper_s8(NULL, &params, NULL, &shape, NULL, &shape, NULL, NULL, NULL, NULL, NULL));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_1_x_n_s8(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_convolve_1_x_n_s8(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, &bias, NULL, NULL));
    #if !defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_convolve_s8(NULL, &params, NULL, NULL, NULL, NULL, NULL, NULL, &bias, NULL, NULL, NULL));
    #else
    (void)bias;
    #endif
    cmsis_nn_dims dims = {6, 2, 1, 9};
    int8_t buffer[600] = {0};
    TEST_ASSERT_EQUAL_INT32(256, arm_convolve_s8_packed_get_filter_size(&dims, 2));
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(NULL, 1));
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, 0));
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, -1));
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(&dims, NULL, 2, buffer));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(&dims, buffer, 2, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(NULL, buffer, 2, buffer + 300));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(&dims, buffer, 2, buffer));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(&dims, buffer, 2, buffer + 107));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_convolve_s8_pack_weights(&dims, buffer + 255, 2, buffer));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_convolve_s8_pack_weights(&dims, buffer, 2, buffer + 108));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_convolve_s8_pack_weights(&dims, buffer + 256, 2, buffer));
    for (int i = 0; i < 4; i++)
    {
        cmsis_nn_dims invalid = dims;
        switch (i)
        {
        case 0:
            invalid.n = 0;
            break;
        case 1:
            invalid.h = -1;
            break;
        case 2:
            invalid.w = 0;
            break;
        case 3:
            invalid.c = -1;
            break;
        }
        TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&invalid, 1));
    }
    dims = (cmsis_nn_dims){INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, 1));
    dims = (cmsis_nn_dims){INT32_MAX, 1, 1, 1};
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, INT32_MAX));
    dims = (cmsis_nn_dims){1, 1, 1, INT32_MAX};
    TEST_ASSERT_EQUAL_INT32(0, arm_convolve_s8_packed_get_filter_size(&dims, 1));
#endif
}

void packed_convolve_s8_matrix_tails(void)
{
#if ARM_NN_ENABLE_AOT && defined(ARM_MATH_MVEI)
    const int32_t reductions[] = {1, 7, 15, 16, 17, 31, 32, 33};
    for (unsigned int r = 0; r < sizeof(reductions) / sizeof(reductions[0]); r++)
    {
        const int32_t cols = reductions[r];
        for (int32_t channels = 1; channels <= 9; channels++)
        {
            cmsis_nn_dims dims = {channels, 1, 1, cols};
            int8_t weights[9 * 33];
            int8_t input[5 * 38];
            int32_t bias[9], multiplier[9], shift[9];
            for (int32_t i = 0; i < channels * cols; i++)
            {
                weights[i] = (int8_t)(i * 17 % 31 - 15);
            }
            for (unsigned int i = 0; i < sizeof(input); i++)
            {
                input[i] = (int8_t)(i * 7 % 41 - 20);
            }
            for (int32_t i = 0; i < channels; i++)
            {
                bias[i] = 137 * (i - 4);
                multiplier[i] = 1073741824 + i * 1000000;
                shift[i] = -3 + i % 3;
            }
            int8_t *packed = malloc(arm_convolve_s8_packed_get_filter_size(&dims, 1));
            TEST_ASSERT_NOT_NULL(packed);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_convolve_s8_pack_weights(&dims, weights, 1, packed));
            for (int32_t positions = 1; positions <= 5; positions++)
            {
                int8_t output[5 * 12 + 2];
                memset(output, 0x55, sizeof(output));
                const int32_t stride = channels + 3;
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                                  arm_nn_mat_mult_nt_t_s8_packed(input,
                                                                 packed,
                                                                 bias,
                                                                 output + 1,
                                                                 multiplier,
                                                                 shift,
                                                                 positions,
                                                                 channels,
                                                                 cols,
                                                                 -7,
                                                                 -101,
                                                                 93,
                                                                 stride,
                                                                 cols + 5));
                TEST_ASSERT_EQUAL_INT8(0x55, output[0]);
                for (int32_t pos = 0; pos < positions; pos++)
                {
                    for (int32_t oc = 0; oc < channels; oc++)
                    {
                        int32_t acc = bias[oc];
                        for (int32_t k = 0; k < cols; k++)
                        {
                            acc += input[pos * (cols + 5) + k] * weights[oc * cols + k];
                        }
                        acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]) - 7;
                        TEST_ASSERT_EQUAL_INT8(MAX(-101, MIN(93, acc)), output[1 + pos * stride + oc]);
                    }
                    for (int32_t oc = channels; oc < stride; oc++)
                    {
                        TEST_ASSERT_EQUAL_INT8(0x55, output[1 + pos * stride + oc]);
                    }
                }
                TEST_ASSERT_EQUAL_INT8(0x55, output[1 + positions * stride]);
            }
            free(packed);
        }
    }
#endif
}

void packed_convolve_s8_group_tails(void)
{
#if ARM_NN_ENABLE_AOT && defined(ARM_MATH_MVEI)
    /* Fast 1x1, strided 1x1, then padded/dilated general convolution.
     * An odd nine-position output also exercises leftover im2col batches.
     */
    for (int32_t mode = 0; mode < 3; mode++)
    {
        for (int32_t groups = 1; groups <= 3; groups++)
        {
            for (int32_t channels = 1; channels <= 7; channels++)
            {
                cmsis_nn_dims input_dims = {1, 5, 5, groups * 17};
                cmsis_nn_dims filter_dims = {groups * channels, mode == 2 ? 2 : 1, mode == 2 ? 2 : 1, 17};
                cmsis_nn_dims output_dims = {1, mode == 0 ? 5 : 3, mode == 0 ? 5 : 3, groups * channels};
                cmsis_nn_dims bias_dims = {1, 1, 1, output_dims.c};
                cmsis_nn_conv_params params = {0};
                params.input_offset = mode == 1 ? -19 : 13;
                params.output_offset = -9;
                params.stride.h = params.stride.w = mode == 0 ? 1 : 2;
                params.dilation.h = params.dilation.w = mode == 2 ? 2 : 1;
                params.padding.h = params.padding.w = mode == 2 ? 1 : 0;
                params.activation.min = -103;
                params.activation.max = 87;
                int8_t input[25 * 51], weights[21 * 68], expected[25 * 21];
                int32_t bias[21], multiplier[21], shift[21];
                const int32_t cols = filter_dims.h * filter_dims.w * filter_dims.c;
                for (int32_t i = 0; i < 25 * input_dims.c; i++)
                {
                    input[i] = (int8_t)(i * 11 % 63 - 31);
                }
                for (int32_t i = 0; i < output_dims.c * cols; i++)
                {
                    weights[i] = (int8_t)(i * 19 % 17 - 8);
                }
                for (int32_t i = 0; i < output_dims.c; i++)
                {
                    bias[i] = 71 * (i - 10);
                    multiplier[i] = 1073741824 + i * 2000000;
                    shift[i] = -5 + i % 4;
                }
                cmsis_nn_per_channel_quant_params quant = {multiplier, shift};
                const int32_t *original_bias = channels % 2 ? NULL : bias;
                /* Use an independent scalar oracle, including null bias for every group. */
                for (int32_t y = 0; y < output_dims.h; y++)
                {
                    for (int32_t x = 0; x < output_dims.w; x++)
                    {
                        for (int32_t oc = 0; oc < output_dims.c; oc++)
                        {
                            int32_t acc = original_bias ? original_bias[oc] : 0;
                            for (int32_t ky = 0; ky < filter_dims.h; ky++)
                            {
                                const int32_t iy = y * params.stride.h + ky * params.dilation.h - params.padding.h;
                                for (int32_t kx = 0; kx < filter_dims.w; kx++)
                                {
                                    const int32_t ix = x * params.stride.w + kx * params.dilation.w - params.padding.w;
                                    if (iy < 0 || iy >= input_dims.h || ix < 0 || ix >= input_dims.w)
                                    {
                                        continue;
                                    }
                                    for (int32_t ic = 0; ic < filter_dims.c; ic++)
                                    {
                                        const int32_t input_index = (iy * input_dims.w + ix) * input_dims.c +
                                            (oc / channels) * filter_dims.c + ic;
                                        const int32_t weight_index =
                                            oc * cols + (ky * filter_dims.w + kx) * filter_dims.c + ic;
                                        acc += (input[input_index] + params.input_offset) * weights[weight_index];
                                    }
                                }
                            }
                            acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]) + params.output_offset;
                            expected[(y * output_dims.w + x) * output_dims.c + oc] =
                                (int8_t)MAX(params.activation.min, MIN(params.activation.max, acc));
                        }
                    }
                }
                validate_aot_convolve_s8(&params,
                                         &quant,
                                         &input_dims,
                                         input,
                                         &filter_dims,
                                         weights,
                                         &bias_dims,
                                         original_bias,
                                         &output_dims,
                                         expected);
            }
        }
    }
#endif
}
