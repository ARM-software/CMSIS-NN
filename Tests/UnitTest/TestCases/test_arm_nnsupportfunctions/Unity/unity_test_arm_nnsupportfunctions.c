/*
 * SPDX-FileCopyrightText: Copyright 2026 Ricardo Alves de Souza <ricardoasouz@hotmail.com>
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

#include <stdint.h>

/* Test the optional assembly path on Thumb-2 targets and the C implementation elsewhere. */
#if defined(__ARM_ARCH_ISA_THUMB) && (__ARM_ARCH_ISA_THUMB >= 2)
    #ifndef CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
        #define CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY
    #endif
#endif

#include "arm_nnsupportfunctions.h"
#include "unity.h"

/*
 * Keep this wrapper out of line to expose the zero-shift carry bug.
 * With the original GCC Thumb-2 code, CMP leaves carry set for INT32_MIN.
 * ASRS with shift zero preserves that carry, and ADC incorrectly adds one.
 * Inlining may generate CMN instead, hiding the regression.
 */
__attribute__((noinline, used)) int32_t divide_by_power_of_two_noinline(int32_t value, int32_t exponent)
{
    return arm_nn_divide_by_power_of_two(value, exponent);
}

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}

void test_arm_nn_divide_by_power_of_two_zero_exponent(void)
{
    static volatile int32_t values[] = {INT32_MIN, INT32_MAX, -1, 0, 1};
    volatile int32_t exponent = 0;

    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); i++)
    {
        const int32_t value = values[i];

        TEST_ASSERT_EQUAL_INT32(value, divide_by_power_of_two_noinline(value, exponent));
    }
}

void test_arm_nn_divide_by_power_of_two_neighbouring_exponents(void)
{
    TEST_ASSERT_EQUAL_INT32(-1073741824, arm_nn_divide_by_power_of_two(INT32_MIN, 1));

    TEST_ASSERT_EQUAL_INT32(-536870912, arm_nn_divide_by_power_of_two(INT32_MIN, 2));

    TEST_ASSERT_EQUAL_INT32(-2, arm_nn_divide_by_power_of_two(-3, 1));

    TEST_ASSERT_EQUAL_INT32(2, arm_nn_divide_by_power_of_two(3, 1));
}
