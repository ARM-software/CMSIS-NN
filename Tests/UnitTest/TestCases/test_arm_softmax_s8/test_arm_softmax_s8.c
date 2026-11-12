/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../TestData/softmax/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void softmax_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_DIFF_MIN;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        TEST_ASSERT_TRUE(validate(output, softmax_output_ref, SOFTMAX_DST_SIZE));
    }
}

void softmax_invalid_diff_min_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = 0x7FFFFFFF;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    int8_t *softmax_expect_invalid_output = malloc(SOFTMAX_DST_SIZE);
    for (int i = 0; i < SOFTMAX_DST_SIZE; i++)
    {
        softmax_expect_invalid_output[i] = -128;
    }

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        TEST_ASSERT_TRUE(validate(output, softmax_expect_invalid_output, SOFTMAX_DST_SIZE));
    }
    free(softmax_expect_invalid_output);
}

void softmax_large_sum_arm_softmax_s8(void)
{
    // input_scale = 1/16 and beta = 1. Both rows have a sum of exponentials >= 512, so every output is below half an
    // output step and rounds to -128. The row size is not a multiple of four, so MVE also runs the scalar tail.
    const int32_t num_rows = 2;
    const int32_t row_size = 1003;
    const int32_t mult = 1073741824;
    const int32_t shift = 23;
    const int32_t diff_min = -248;

    int8_t *input_data = malloc(num_rows * row_size);
    int8_t *output = malloc(num_rows * row_size);
    int8_t *expected = malloc(num_rows * row_size);

    for (int i = 0; i < row_size; i++)
    {
        input_data[i] = 0;
        input_data[row_size + i] = -4;
    }
    input_data[row_size + 7] = 4;
    memset(expected, -128, num_rows * row_size);

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        TEST_ASSERT_TRUE(validate(output, expected, num_rows * row_size));
    }
    free(input_data);
    free(output);
    free(expected);
}
