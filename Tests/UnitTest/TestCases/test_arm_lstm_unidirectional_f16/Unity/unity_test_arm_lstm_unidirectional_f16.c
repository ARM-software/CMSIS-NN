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

#include "../test_arm_lstm_unidirectional_f16.c"
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
void test_lstm_small_f16_arm_lstm_unidirectional_f16(void) { lstm_small_f16_arm_lstm_unidirectional_f16(); }
void test_lstm_medium_f16_arm_lstm_unidirectional_f16(void) { lstm_medium_f16_arm_lstm_unidirectional_f16(); }
void test_lstm_large_f16_arm_lstm_unidirectional_f16(void) { lstm_large_f16_arm_lstm_unidirectional_f16(); }
void test_lstm_match_1_f16_arm_lstm_unidirectional_f16(void) { lstm_match_1_f16_arm_lstm_unidirectional_f16(); }
void test_lstm_match_2_f16_arm_lstm_unidirectional_f16(void) { lstm_match_2_f16_arm_lstm_unidirectional_f16(); }
void test_lstm_match_one_time_step_f16_arm_lstm_unidirectional_f16(void)
{
    lstm_match_one_time_step_f16_arm_lstm_unidirectional_f16();
}
