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

#include <stdint.h>

/*
 * The board device-definition layer expects a sleep hook for a few peripheral
 * descriptors. The unit tests do not exercise those timed peripherals, so a
 * coarse busy-wait is sufficient here.
 */
__attribute__((weak)) void wait_us(uint32_t usec)
{
    while (usec-- != 0U)
    {
        for (volatile uint32_t i = 0; i < 25U; ++i)
        {
            __asm volatile("nop");
        }
    }
}
