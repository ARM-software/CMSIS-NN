/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "validate.h"

static inline void validate_aot_convolve_s8(const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            const int8_t *output_ref)
{
#if ARM_NN_ENABLE_AOT && defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params aot_params = *conv_params;
    aot_params.filter_format = ARM_NN_FILTER_FORMAT_PACKED_4X16;
    aot_params.bias_format = ARM_NN_BIAS_FORMAT_EFFECTIVE;
    conv_params = &aot_params;
    const int32_t output_size = input_dims->n * output_dims->h * output_dims->w * output_dims->c;
    const int32_t rhs_cols = filter_dims->h * filter_dims->w * filter_dims->c;
    const int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
    cmsis_nn_dims packed_dims = *filter_dims;
    packed_dims.n = output_dims->c;
    const int32_t groups = input_dims->c / filter_dims->c;
    const int32_t packed_size = arm_convolve_s8_packed_get_filter_size(&packed_dims, groups);
    int8_t *packed_filter = malloc(packed_size);
    TEST_ASSERT_NOT_NULL(packed_filter);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_s8_pack_weights(&packed_dims, filter_data, groups, packed_filter));
    int32_t *effective_bias = malloc(output_dims->c * sizeof(int32_t));
    int8_t *guarded_output = malloc(output_size + 2);
    int8_t *scratch = malloc(buf_size > 0 ? buf_size : 1);
    TEST_ASSERT_NOT_NULL(effective_bias);
    TEST_ASSERT_NOT_NULL(guarded_output);
    TEST_ASSERT_NOT_NULL(scratch);
    int8_t *output = guarded_output + 1;
    cmsis_nn_context ctx = {scratch, (uint32_t)buf_size};

    memset(guarded_output, 0x55, output_size + 2);
    memset(output, 0, output_size * sizeof(int8_t));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_vector_sum_s8(
            effective_bias, rhs_cols, output_dims->c, filter_data, conv_params->input_offset, 0, bias_data));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_s8(&ctx,
                                              conv_params,
                                              quant_params,
                                              input_dims,
                                              input_data,
                                              filter_dims,
                                              packed_filter,
                                              bias_dims,
                                              effective_bias,
                                              output_dims,
                                              output));
    TEST_ASSERT_TRUE(validate(output, output_ref, output_size));
    TEST_ASSERT_EQUAL_INT8(0x55, guarded_output[0]);
    TEST_ASSERT_EQUAL_INT8(0x55, output[output_size]);
    if (input_dims->h == 1 && filter_dims->h == 1 && output_dims->h == 1)
    {
        cmsis_nn_conv_params legacy_params = aot_params;
        legacy_params.filter_format = ARM_NN_FILTER_FORMAT_ROW_MAJOR;
        legacy_params.bias_format = ARM_NN_BIAS_FORMAT_ORIGINAL;
        TEST_ASSERT_EQUAL_INT32(
            arm_convolve_1_x_n_s8_get_buffer_size(&legacy_params, input_dims, filter_dims, output_dims),
            arm_convolve_1_x_n_s8_get_buffer_size(conv_params, input_dims, filter_dims, output_dims));
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_NO_IMPL_ERROR,
            arm_convolve_1_x_n_s8(NULL, conv_params, NULL, NULL, NULL, NULL, NULL, NULL, effective_bias, NULL, NULL));
    }
    free(packed_filter);

    memset(ctx.buf, 0, ctx.size);
    free(scratch);
    free(guarded_output);
    free(effective_bias);
#else
    (void)conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input_data;
    (void)filter_dims;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_dims;
    (void)output_ref;

#endif
}
