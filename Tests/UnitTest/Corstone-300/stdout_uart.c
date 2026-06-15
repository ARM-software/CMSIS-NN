/*
 * Copyright (c) 2026 Arm Limited. All rights reserved.
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

#include "uart.h"

static int stdout_initialized;

/*
 * Bring up the UART on first use so stdio output works even if no explicit
 * board-level stdout init was performed before the test starts.
 */
static int stdout_ensure_initialized(void)
{
    if (!stdout_initialized)
    {
        uart_init();
        stdout_initialized = 1;
    }

    return 0;
}

int stdout_putchar(int ch)
{
    if (stdout_ensure_initialized() != 0)
    {
        return -1;
    }

    return uart_putc((unsigned char)ch);
}

int stderr_putchar(int ch) { return stdout_putchar(ch); }
