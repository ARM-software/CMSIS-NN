/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_activation_f16.c"
#include "unity.h"

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
void test_activation_f16_arm_nn_activation_f16_sigmoid(void) { activation_f16_arm_nn_activation_f16_sigmoid(); }
void test_activation_f16_arm_nn_activation_f16_tanh(void) { activation_f16_arm_nn_activation_f16_tanh(); }
void test_activation_f16_arm_nn_activation_f16_hardswish(void) { activation_f16_arm_nn_activation_f16_hardswish(); }
void test_activation_f16_arm_nn_activation_f16_leaky_relu(void) { activation_f16_arm_nn_activation_f16_leaky_relu(); }
