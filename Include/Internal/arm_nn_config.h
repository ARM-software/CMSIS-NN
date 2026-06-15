/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_config.h
 * Description:  Internal optional feature configuration for CMSIS-NN
 *
 * $Date:        24 April 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * This header is part of the internal CMSIS-NN build configuration surface.
 * Applications are expected to include the public arm_nnfunctions*.h headers,
 * which pull the needed configuration through the public type/math headers.
 *
 * The current options focus on floating-point support, but the file name is
 * kept generic so wider CMSIS-NN configuration can be added here over time,
 * including settings shared with quantized paths.
 */

#ifndef ARM_NN_CONFIG_H
#define ARM_NN_CONFIG_H

/**
 *
 * @brief Optional feature gates for floating-point extensions
 *
 * These are disabled by default so integer-only builds keep their current
 * code size and API surface. The float16 feature gate still depends on
 * toolchain and target support such as ARM_FLOAT16_SUPPORTED where
 * applicable.
 *
 */

#ifndef ARM_NN_ENABLE_F32
    #define ARM_NN_ENABLE_F32 0
#endif

#ifndef ARM_NN_ENABLE_F16
    #define ARM_NN_ENABLE_F16 0
#endif

#define ARM_NN_FLOAT_API_ENABLED (ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16)

/*
 * NN_DISABLE_SPECIALIZATION disables optional shape/layout-specific fast paths
 * and forces the corresponding generic implementations. This is useful for
 * debugging or validating specialized kernels against the generic path.
 */

/**
 *
 * @brief Optional floating-point softmax exp approximation selection
 *
 * Define one of these macros at build time to select the scalar float softmax
 * exp path:
 *   - ARM_NN_USE_EXP_LUT
 *   - ARM_NN_USE_EXP_TAYLOR
 *
 * If neither macro is defined, the scalar softmax path defaults to the LUT
 * implementation for performance. This adds one 257-entry lookup table per
 * enabled float precision:
 *   - float16: 257 half-words, about 514 bytes
 *   - float32: 257 words, about 1028 bytes
 *
 * Define ARM_NN_USE_EXP_TAYLOR to avoid the extra lookup-table storage.
 *
 */

#if defined(ARM_NN_USE_EXP_LUT) && defined(ARM_NN_USE_EXP_TAYLOR)
    #error "ARM_NN_USE_EXP_LUT and ARM_NN_USE_EXP_TAYLOR are mutually exclusive"
#endif

#endif /* ARM_NN_CONFIG_H */
