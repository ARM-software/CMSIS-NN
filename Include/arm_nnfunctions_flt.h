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
 * Title:        arm_nnfunctions_flt.h
 * Description:  Public floating-point API extensions for CMSIS-NN
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NNFUNCTIONS_FLT_H
#define ARM_NNFUNCTIONS_FLT_H

#include "arm_nn_types_flt.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Floating-point public APIs.
 * Float operators currently use NHWC layout.
 */

#if ARM_NN_ENABLE_F32

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Depthwise convolution, NHWC layout.
 *
 * @param[in,out] ctx            Function context that may hold a temporary scratch buffer.
 * @param[in]     dw_conv_params Depthwise convolution parameters (stride, padding, dilation, channel multiplier and
 *                               activation clamp).
 * @param[in]     input_dims     Input tensor dimensions in NHWC format.
 * @param[in]     input          Pointer to the input tensor data.
 * @param[in]     filter_dims    Filter tensor dimensions in NHWC-compatible depthwise format.
 * @param[in]     kernel         Pointer to the filter tensor data.
 * @param[in]     bias_dims      Bias tensor dimensions. Format: [C_OUT].
 * @param[in]     bias           Optional bias tensor data.
 * @param[in]     output_dims    Output tensor dimensions in NHWC format.
 * @param[out]    output         Pointer to the output tensor data.
 *
 * @note When @p ctx->buf is used for internal kernel repacking, it must be aligned to the element type stored in
 * scratch: at least 4-byte aligned for float32_t and, via @copydoc, at least 2-byte aligned for float16_t.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_depthwise_nhwc_conv_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output);

/**
 * @brief Depthwise convolution, dispatch by layout.
 *
 * @param[in,out] ctx            Function context that may hold a temporary scratch buffer.
 * @param[in]     dw_conv_params Depthwise convolution parameters (stride, padding, dilation, channel multiplier and
 *                               activation clamp).
 * @param[in]     input_dims     Input tensor dimensions. Format depends on @p layout.
 * @param[in]     input          Pointer to the input tensor data.
 * @param[in]     filter_dims    Filter tensor dimensions. Format depends on @p layout.
 * @param[in]     kernel         Pointer to the filter tensor data.
 * @param[in]     bias_dims      Bias tensor dimensions. Format: [C_OUT].
 * @param[in]     bias           Optional bias tensor data.
 * @param[in]     output_dims    Output tensor dimensions. Format depends on @p layout.
 * @param[out]    output         Pointer to the output tensor data.
 * @param[in]     layout         Tensor layout selector. Current float APIs require `ARM_NN_LAYOUT_NHWC`.
 *
 * @note When @p ctx->buf is used for internal kernel repacking, it must be aligned to the element type stored in
 * scratch: at least 4-byte aligned for float32_t and, via @copydoc, at least 2-byte aligned for float16_t.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_depthwise_conv_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output,
                                           arm_nn_tensor_layout layout);

/**
 * @brief Depthwise convolution wrapper using the CMSIS-NN baseline path.
 *
 * @param[in,out] ctx            Function context that may hold a temporary scratch buffer.
 * @param[in]     dw_conv_params Depthwise convolution parameters.
 * @param[in]     input_dims     Input tensor dimensions.
 * @param[in]     input          Pointer to the input tensor data.
 * @param[in]     filter_dims    Filter tensor dimensions.
 * @param[in]     kernel         Pointer to the filter tensor data.
 * @param[in]     bias_dims      Bias tensor dimensions. Format: [C_OUT].
 * @param[in]     bias           Optional bias tensor data.
 * @param[in]     output_dims    Output tensor dimensions.
 * @param[out]    output         Pointer to the output tensor data.
 *
 * @note When @p ctx->buf is used for internal kernel repacking, it must be aligned to the element type stored in
 * scratch: at least 4-byte aligned for float32_t and, via @copydoc, at least 2-byte aligned for float16_t.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_f32(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float32_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float32_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float32_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float32_t *output);

/**
 * @brief Get the temporary buffer size required by depthwise convolution.
 *
 * @param[in] dw_conv_params Depthwise convolution parameters.
 * @param[in] input_dims     Input tensor dimensions.
 * @param[in] filter_dims    Filter tensor dimensions.
 * @param[in] output_dims    Output tensor dimensions.
 * @param[in] layout         Tensor layout selector.
 *
 * @return Required buffer size in bytes, or 0 when no scratch buffer is needed.
 */
int32_t arm_depthwise_conv_f32_get_buffer_size(const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

/**
 * @brief Get the buffer size required by the depthwise convolution wrapper.
 */
int32_t arm_depthwise_conv_wrapper_f32_get_buffer_size(const cmsis_nn_dw_conv_params_f32 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

/**
 * @brief Convolution, NHWC layout.
 */
arm_cmsis_nn_status arm_convolve_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f32 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float32_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *output_data);

/**
 * @brief Convolution, dispatch by layout.
 *
 * @param[in,out] ctx         Function context that may hold a temporary scratch buffer.
 * @param[in]     conv_params Convolution parameters (stride, padding, dilation and activation clamp).
 * @param[in]     input_dims  Input tensor dimensions. Format depends on @p layout.
 * @param[in]     input_data  Pointer to the input tensor data.
 * @param[in]     filter_dims Filter tensor dimensions. Format depends on @p layout.
 * @param[in]     filter_data Pointer to the filter tensor data.
 * @param[in]     bias_dims   Bias tensor dimensions. Format: [C_OUT].
 * @param[in]     bias_data   Optional bias tensor data.
 * @param[in]     output_dims Output tensor dimensions. Format depends on @p layout.
 * @param[out]    output_data Pointer to the output tensor data.
 * @param[in]     layout      Tensor layout selector. Current float APIs require `ARM_NN_LAYOUT_NHWC`.
 *
 * @note When `conv_params->weight_format` is set to
 *       `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED`, the matmul-backed convolution
 *       paths interpret @p filter_data as an already prepacked `NTxN` RHS
 *       buffer instead of the standard public filter layout.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_convolve_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f32 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float32_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float32_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *output_data,
                                     arm_nn_tensor_layout layout);

/**
 * @brief Convolution wrapper using the CMSIS-NN baseline path.
 */
arm_cmsis_nn_status arm_convolve_wrapper_f32(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float32_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float32_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float32_t *output_data);

/**
 * @brief 1x1 convolution, NHWC layout.
 */
arm_cmsis_nn_status arm_convolve_1x1_nhwc_f32(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f32 *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const float32_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float32_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float32_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float32_t *output_data);

/**
 * @brief 1x1 convolution, dispatch by layout.
 *
 * @note When `conv_params->weight_format` is set to
 *       `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED`, the matmul-backed 1x1
 *       convolution paths interpret @p filter_data as an already prepacked
 *       `NTxN` RHS buffer instead of the standard public filter layout.
 */
arm_cmsis_nn_status arm_convolve_1x1_f32(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params_f32 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const float32_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const float32_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const float32_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         float32_t *output_data,
                                         arm_nn_tensor_layout layout);

/**
 * @brief 1xN convolution, NHWC layout.
 */
arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_conv_params_f32 *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output_data);

/**
 * @brief 1xN convolution, dispatch by layout.
 */
arm_cmsis_nn_status arm_convolve_1_x_n_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_conv_params_f32 *conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output_data,
                                           arm_nn_tensor_layout layout);

/**
 * @brief Get the temporary buffer size required by convolution.
 *
 * @param[in] conv_params  Convolution parameters.
 * @param[in] input_dims   Input tensor dimensions.
 * @param[in] filter_dims  Filter tensor dimensions.
 * @param[in] output_dims  Output tensor dimensions.
 * @param[in] layout       Tensor layout selector.
 *
 * @note When `conv_params->weight_format` is
 *       `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED`, this still reports only the
 *       temporary input/im2col scratch requirement. Any offline-packed
 *       filter storage is expected to be provided by the caller.
 *
 * @return Required buffer size in bytes, or 0 when no scratch buffer is needed.
 */
int32_t arm_convolve_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout);

/**
 * @brief Get the buffer size required by the convolution wrapper.
 */
int32_t arm_convolve_wrapper_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

/**
 * @brief Get the buffer size required by 1x1 convolution.
 *
 * @note Returns `0` for the unity-stride no-pack path. For non-unity-stride NHWC
 *       1x1 convolution, the returned scratch size enables the packed-tile + GEMM path.
 *       When `conv_params->weight_format` is `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED`,
 *       this still excludes the offline-packed filter storage itself.
 */
int32_t arm_convolve_1x1_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             arm_nn_tensor_layout layout);

/**
 * @brief Get the buffer size required by 1xN convolution.
 */
int32_t arm_convolve_1_x_n_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

/** @} */

/**
 * @addtogroup Pooling
 * @{
 */

/**
 * @brief Max pooling.
 *
 * @param[in,out] ctx         Function context that may hold a temporary scratch buffer.
 * @param[in]     pool_params Pooling parameters (stride, padding and activation clamp).
 * @param[in]     input_dims  Input tensor dimensions.
 * @param[in]     src         Pointer to the input tensor data.
 * @param[in]     filter_dims Pooling kernel dimensions.
 * @param[in]     output_dims Output tensor dimensions.
 * @param[out]    dst         Pointer to the output tensor data.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_max_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst);

/**
 * @brief Average pooling.
 *
 * @param[in,out] ctx         Function context that may hold a temporary scratch buffer.
 * @param[in]     pool_params Pooling parameters (stride, padding and activation clamp).
 * @param[in]     input_dims  Input tensor dimensions.
 * @param[in]     src         Pointer to the input tensor data.
 * @param[in]     filter_dims Pooling kernel dimensions.
 * @param[in]     output_dims Output tensor dimensions.
 * @param[out]    dst         Pointer to the output tensor data.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_avg_pool_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f32 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float32_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float32_t *dst);

/** @} */

/**
 * @addtogroup Acti
 * @{
 */

/**
 * @brief Elementwise activation.
 *
 * @param[in]  input     Pointer to the input samples.
 * @param[out] output    Pointer to the output samples.
 * @param[in]  size      Number of elements to process.
 * @param[in]  type      Activation selector.
 * @param[in]  act_param Extra activation parameter. Used for parameterized activations such as leaky ReLU.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_activation_f32(const float32_t *input,
                                          float32_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float32_t act_param);

/** @} */

/**
 * @addtogroup groupElementwise
 * @{
 */

/**
 * @brief Elementwise add with optional output clamp.
 *
 * @param[in]  input_1_vect        Pointer to the first input vector.
 * @param[in]  input_2_vect        Pointer to the second input vector.
 * @param[out] output              Pointer to the output vector.
 * @param[in]  out_activation_min  Minimum output clamp value.
 * @param[in]  out_activation_max  Maximum output clamp value.
 * @param[in]  block_size          Number of elements to process.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_elementwise_add_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

/**
 * @brief Elementwise multiply with optional output clamp.
 *
 * @param[in]  input_1_vect        Pointer to the first input vector.
 * @param[in]  input_2_vect        Pointer to the second input vector.
 * @param[out] output              Pointer to the output vector.
 * @param[in]  out_activation_min  Minimum output clamp value.
 * @param[in]  out_activation_max  Maximum output clamp value.
 * @param[in]  block_size          Number of elements to process.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_elementwise_mul_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size);

/**
 * @brief Elementwise minimum.
 */
arm_cmsis_nn_status arm_minimum_f32(const cmsis_nn_context *ctx,
                                    const float32_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims);

/**
 * @brief Elementwise maximum.
 */
arm_cmsis_nn_status arm_maximum_f32(const cmsis_nn_context *ctx,
                                    const float32_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float32_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float32_t *output_data,
                                    const cmsis_nn_dims *output_dims);

/** @} */

/**
 * @addtogroup FC
 * @{
 */

/**
 * @brief Fully connected layer, NHWC layout.
 */
arm_cmsis_nn_status arm_fully_connected_nhwc_f32(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_fc_params_f32 *fc_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float32_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float32_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float32_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float32_t *output);

/**
 * @brief Fully connected layer, dispatch by layout.
 *
 * @param[in,out] ctx         Function context that may hold a temporary scratch buffer.
 * @param[in]     fc_params   Fully connected parameters and activation clamp.
 * @param[in]     input_dims  Input tensor dimensions.
 * @param[in]     input       Pointer to the input tensor data.
 * @param[in]     filter_dims Filter tensor dimensions.
 * @param[in]     kernel      Pointer to the filter tensor data.
 * @param[in]     bias_dims   Bias tensor dimensions.
 * @param[in]     bias        Optional bias tensor data.
 * @param[in]     output_dims Output tensor dimensions.
 * @param[out]    output      Pointer to the output tensor data.
 * @param[in]     layout      Tensor layout selector. Current float APIs require `ARM_NN_LAYOUT_NHWC`.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_fully_connected_f32(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params_f32 *fc_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float32_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const float32_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const float32_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            float32_t *output,
                                            arm_nn_tensor_layout layout);

/**
 * @brief Get the temporary buffer size required by the fully connected layer.
 *
 * @param[in] fc_params   Fully connected parameters.
 * @param[in] input_dims  Input tensor dimensions.
 * @param[in] filter_dims Filter tensor dimensions.
 * @param[in] output_dims Output tensor dimensions.
 * @param[in] layout      Tensor layout selector.
 *
 * @return Required buffer size in bytes, or 0 when no scratch buffer is needed.
 */
int32_t arm_fully_connected_f32_get_buffer_size(const cmsis_nn_fc_params_f32 *fc_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                arm_nn_tensor_layout layout);

/** @} */

/**
 * @addtogroup NNSupport
 * @{
 */

/**
 * @brief Transpose a floating-point tensor.
 *
 * @param[in,out] ctx         Function context that may hold a temporary scratch buffer.
 * @param[in]     params      Transpose parameters, including permutation and layout information.
 * @param[in]     input_dims  Input tensor dimensions.
 * @param[in]     input       Pointer to the input tensor data.
 * @param[in]     output_dims Output tensor dimensions.
 * @param[out]    output      Pointer to the output tensor data.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_transpose_f32(const cmsis_nn_context *ctx,
                                      const cmsis_nn_transpose_params_f32 *params,
                                      const cmsis_nn_dims *input_dims,
                                      const float32_t *input,
                                      const cmsis_nn_dims *output_dims,
                                      float32_t *output);

/**
 * @brief Concatenate tensors along the X axis.
 */
void arm_concatenation_f32_x(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_x,
                             uint32_t offset_x);

/**
 * @brief Concatenate tensors along the Y axis.
 */
void arm_concatenation_f32_y(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_y,
                             uint32_t offset_y);

/**
 * @brief Concatenate tensors along the Z axis.
 */
void arm_concatenation_f32_z(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             int32_t output_z,
                             uint32_t offset_z);

/**
 * @brief Concatenate tensors along the W axis.
 */
void arm_concatenation_f32_w(const float32_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float32_t *output,
                             uint32_t offset_w);

/** @} */

/**
 * @addtogroup Pad
 * @{
 */

/**
 * @brief Pad a tensor with a constant value.
 */
arm_cmsis_nn_status arm_pad_f32(const float32_t *input,
                                float32_t *output,
                                float32_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad);

/** @} */

/**
 * @addtogroup NNSupport
 * @{
 */

/**
 * @brief Apply batch normalization.
 */
arm_cmsis_nn_status arm_batch_norm_f32(const float32_t *input,
                                       float32_t *output,
                                       const float32_t *scale,
                                       const float32_t *bias,
                                       const cmsis_nn_dims *input_dims,
                                       arm_nn_tensor_layout layout);

/**
 * @brief Reshape by copying data without changing element order.
 */
void arm_reshape_f32(const float32_t *input, float32_t *output, uint32_t total_size);

/** @} */

/**
 * @addtogroup FC
 * @{
 */

/**
 * @brief Batched matrix multiplication.
 *
 * @param[in,out] ctx            Function context that may hold a temporary scratch buffer.
 * @param[in]     bmm_params     Batch matmul parameters and activation clamp.
 * @param[in]     input_lhs_dims Left-hand-side input tensor dimensions.
 * @param[in]     input_lhs      Pointer to the left-hand-side input tensor.
 * @param[in]     input_rhs_dims Right-hand-side input tensor dimensions.
 * @param[in]     input_rhs      Pointer to the right-hand-side input tensor.
 * @param[in]     output_dims    Output tensor dimensions.
 * @param[out]    output         Pointer to the output tensor.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_batch_matmul_f32(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params_f32 *bmm_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const float32_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const float32_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         float32_t *output);

/**
 * @brief Get the temporary buffer size required by batched matrix multiplication.
 *
 * @param[in] bmm_params     Batch matmul parameters.
 * @param[in] input_lhs_dims Left-hand-side input tensor dimensions.
 * @param[in] input_rhs_dims Right-hand-side input tensor dimensions.
 * @param[in] output_dims    Output tensor dimensions.
 *
 * @return Required buffer size in bytes, or 0 when no scratch buffer is needed.
 */
int32_t arm_batch_matmul_f32_get_buffer_size(const cmsis_nn_bmm_params_f32 *bmm_params,
                                             const cmsis_nn_dims *input_lhs_dims,
                                             const cmsis_nn_dims *input_rhs_dims,
                                             const cmsis_nn_dims *output_dims);

/** @} */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Transpose convolution wrapper using the CMSIS-NN baseline path.
 */
arm_cmsis_nn_status arm_transpose_conv_wrapper_f32(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_context *output_ctx,
                                                   const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float32_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float32_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float32_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   float32_t *output_data,
                                                   arm_nn_tensor_layout layout);

/**
 * @brief Transpose convolution, NHWC layout.
 */
arm_cmsis_nn_status arm_transpose_conv_nhwc_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_context *output_ctx,
                                                const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output_data);

/**
 * @brief Transpose convolution, dispatch by layout.
 *
 * @param[in,out] ctx                   Function context that may hold a temporary scratch buffer.
 * @param[in,out] output_ctx            Output accumulation context for helper implementations.
 * @param[in]     transpose_conv_params Transpose convolution parameters.
 * @param[in]     input_dims            Input tensor dimensions.
 * @param[in]     input_data            Pointer to the input tensor data.
 * @param[in]     filter_dims           Filter tensor dimensions.
 * @param[in]     filter_data           Pointer to the filter tensor data.
 * @param[in]     bias_dims             Bias tensor dimensions.
 * @param[in]     bias_data             Optional bias tensor data.
 * @param[in]     output_dims           Output tensor dimensions.
 * @param[out]    output_data           Pointer to the output tensor data.
 * @param[in]     layout                Tensor layout selector. Current float APIs require `ARM_NN_LAYOUT_NHWC`.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_transpose_conv_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output_data,
                                           arm_nn_tensor_layout layout);

/**
 * @brief Get the temporary buffer size required by transpose convolution.
 *
 * @param[in] transpose_conv_params Transpose convolution parameters.
 * @param[in] input_dims            Input tensor dimensions.
 * @param[in] filter_dims           Filter tensor dimensions.
 * @param[in] out_dims              Output tensor dimensions.
 *
 * @return Required buffer size in bytes, or 0 when no scratch buffer is needed.
 */
int32_t arm_transpose_conv_f32_get_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims);

/**
 * @brief Get the reverse-convolution workspace size used by transpose convolution helpers.
 *
 * @param[in] transpose_conv_params Transpose convolution parameters.
 * @param[in] input_dims            Input tensor dimensions.
 * @param[in] filter_dims           Filter tensor dimensions.
 *
 * @return Required buffer size in bytes, or 0 when no reverse-convolution buffer is needed.
 */
int32_t
arm_transpose_conv_f32_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims);

/** @} */

/**
 * @addtogroup SVDF
 * @{
 */

/**
 * @brief Stateful singular value decomposition filter.
 *
 * @param[in,out] ctx                  Function context that may hold a temporary scratch buffer.
 * @param[in,out] input_ctx            Optional input staging context.
 * @param[in,out] output_ctx           Optional output staging context.
 * @param[in]     svdf_params          SVDF operator parameters.
 * @param[in]     input_dims           Input tensor dimensions.
 * @param[in]     input_data           Pointer to the input tensor data.
 * @param[in]     state_dims           State tensor dimensions.
 * @param[in,out] state_data           Pointer to the mutable state tensor.
 * @param[in]     weights_feature_dims Feature-weight tensor dimensions.
 * @param[in]     weights_feature_data Pointer to the feature-weight tensor.
 * @param[in]     weights_time_dims    Time-weight tensor dimensions.
 * @param[in]     weights_time_data    Pointer to the time-weight tensor.
 * @param[in]     bias_dims            Bias tensor dimensions.
 * @param[in]     bias_data            Optional bias tensor data.
 * @param[in]     output_dims          Output tensor dimensions.
 * @param[out]    output_data          Pointer to the output tensor data.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_svdf_f32(const cmsis_nn_context *ctx,
                                 const cmsis_nn_context *input_ctx,
                                 const cmsis_nn_context *output_ctx,
                                 const cmsis_nn_svdf_params_f32 *svdf_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float32_t *input_data,
                                 const cmsis_nn_dims *state_dims,
                                 float32_t *state_data,
                                 const cmsis_nn_dims *weights_feature_dims,
                                 const float32_t *weights_feature_data,
                                 const cmsis_nn_dims *weights_time_dims,
                                 const float32_t *weights_time_data,
                                 const cmsis_nn_dims *bias_dims,
                                 const float32_t *bias_data,
                                 const cmsis_nn_dims *output_dims,
                                 float32_t *output_data);

/** @} */

/**
 * @addtogroup LSTM
 * @{
 */

/**
 * @brief Unidirectional LSTM inference.
 *
 * @param[in]     input   Pointer to the input sequence tensor.
 * @param[out]    output  Pointer to the output sequence tensor.
 * @param[in]     params  LSTM parameters and weights.
 * @param[in,out] buffers Mutable LSTM scratch and state buffers.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_lstm_unidirectional_f32(const float32_t *input,
                                                float32_t *output,
                                                const cmsis_nn_lstm_params_f32 *params,
                                                cmsis_nn_lstm_context_f32 *buffers);

/** @} */

/**
 * @addtogroup Softmax
 * @{
 */

/**
 * @brief Softmax using the float-native API signature.
 *
 * @param[in]  input    Pointer to the input matrix stored as @p num_rows rows of @p row_size values.
 * @param[in]  num_rows Number of rows in the input matrix.
 * @param[in]  row_size Number of columns per row.
 * @param[out] output   Pointer to the output matrix.
 *
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_softmax_f32(const float32_t *input, int32_t num_rows, int32_t row_size, float32_t *output);

    /** @} */

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @copydoc arm_depthwise_nhwc_conv_f32
 */
arm_cmsis_nn_status arm_depthwise_nhwc_conv_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output);

/**
 * @copydoc arm_depthwise_conv_f32
 */
arm_cmsis_nn_status arm_depthwise_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output,
                                           arm_nn_tensor_layout layout);

/**
 * @copydoc arm_depthwise_conv_wrapper_f32
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output);

/**
 * @copydoc arm_depthwise_conv_f32_get_buffer_size
 */
int32_t arm_depthwise_conv_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

/**
 * @copydoc arm_depthwise_conv_wrapper_f32_get_buffer_size
 */
int32_t arm_depthwise_conv_wrapper_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims);

/**
 * @copydoc arm_convolve_nhwc_f32
 */
arm_cmsis_nn_status arm_convolve_nhwc_f16(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f16 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float16_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float16_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float16_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float16_t *output_data);

/**
 * @copydoc arm_convolve_f32
 */
arm_cmsis_nn_status arm_convolve_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f16 *conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *input_data,
                                     const cmsis_nn_dims *filter_dims,
                                     const float16_t *filter_data,
                                     const cmsis_nn_dims *bias_dims,
                                     const float16_t *bias_data,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *output_data,
                                     arm_nn_tensor_layout layout);

/**
 * @copydoc arm_convolve_wrapper_f32
 */
arm_cmsis_nn_status arm_convolve_wrapper_f16(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float16_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float16_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float16_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float16_t *output_data);

/**
 * @copydoc arm_convolve_1x1_nhwc_f32
 */
arm_cmsis_nn_status arm_convolve_1x1_nhwc_f16(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f16 *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const float16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float16_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float16_t *output_data);

/**
 * @copydoc arm_convolve_1x1_f32
 */
arm_cmsis_nn_status arm_convolve_1x1_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const float16_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const float16_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const float16_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output_data,
                                         arm_nn_tensor_layout layout);

/**
 * @copydoc arm_convolve_1_x_n_nhwc_f32
 */
arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_conv_params_f16 *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output_data);

/**
 * @copydoc arm_convolve_1_x_n_f32
 */
arm_cmsis_nn_status arm_convolve_1_x_n_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_conv_params_f16 *conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output_data,
                                           arm_nn_tensor_layout layout);

/**
 * @copydoc arm_convolve_f32_get_buffer_size
 */
int32_t arm_convolve_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout);

/**
 * @copydoc arm_convolve_wrapper_f32_get_buffer_size
 */
int32_t arm_convolve_wrapper_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims);

/**
 * @copydoc arm_convolve_1x1_f32_get_buffer_size
 */
int32_t arm_convolve_1x1_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             arm_nn_tensor_layout layout);

/**
 * @copydoc arm_convolve_1_x_n_f32_get_buffer_size
 */
int32_t arm_convolve_1_x_n_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout);

/** @} */

/**
 * @addtogroup Pooling
 * @{
 */

/**
 * @copydoc arm_max_pool_f32
 */
arm_cmsis_nn_status arm_max_pool_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f16 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *dst);

/**
 * @copydoc arm_avg_pool_f32
 */
arm_cmsis_nn_status arm_avg_pool_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_pool_params_f16 *pool_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *src,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *dst);

/** @} */

/**
 * @addtogroup Acti
 * @{
 */

/**
 * @copydoc arm_nn_activation_f32
 */
arm_cmsis_nn_status arm_nn_activation_f16(const float16_t *input,
                                          float16_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float16_t act_param);

/** @} */

/**
 * @addtogroup groupElementwise
 * @{
 */

/**
 * @copydoc arm_elementwise_add_f32
 */
arm_cmsis_nn_status arm_elementwise_add_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

/**
 * @copydoc arm_elementwise_mul_f32
 */
arm_cmsis_nn_status arm_elementwise_mul_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size);

/**
 * @copydoc arm_minimum_f32
 */
arm_cmsis_nn_status arm_minimum_f16(const cmsis_nn_context *ctx,
                                    const float16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

/**
 * @copydoc arm_maximum_f32
 */
arm_cmsis_nn_status arm_maximum_f16(const cmsis_nn_context *ctx,
                                    const float16_t *input_1_data,
                                    const cmsis_nn_dims *input_1_dims,
                                    const float16_t *input_2_data,
                                    const cmsis_nn_dims *input_2_dims,
                                    float16_t *output_data,
                                    const cmsis_nn_dims *output_dims);

/** @} */

/**
 * @addtogroup FC
 * @{
 */

/**
 * @copydoc arm_fully_connected_nhwc_f32
 */
arm_cmsis_nn_status arm_fully_connected_nhwc_f16(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_fc_params_f16 *fc_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output);

/**
 * @copydoc arm_fully_connected_f32
 */
arm_cmsis_nn_status arm_fully_connected_f16(const cmsis_nn_context *ctx,
                                            const cmsis_nn_fc_params_f16 *fc_params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input,
                                            const cmsis_nn_dims *filter_dims,
                                            const float16_t *kernel,
                                            const cmsis_nn_dims *bias_dims,
                                            const float16_t *bias,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output,
                                            arm_nn_tensor_layout layout);

/**
 * @copydoc arm_fully_connected_f32_get_buffer_size
 */
int32_t arm_fully_connected_f16_get_buffer_size(const cmsis_nn_fc_params_f16 *fc_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                arm_nn_tensor_layout layout);

/** @} */

/**
 * @addtogroup NNSupport
 * @{
 */

/**
 * @copydoc arm_transpose_f32
 */
arm_cmsis_nn_status arm_transpose_f16(const cmsis_nn_context *ctx,
                                      const cmsis_nn_transpose_params_f16 *params,
                                      const cmsis_nn_dims *input_dims,
                                      const float16_t *input,
                                      const cmsis_nn_dims *output_dims,
                                      float16_t *output);

/**
 * @brief arm concatenation f16 x
 */
void arm_concatenation_f16_x(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_x,
                             uint32_t offset_x);

/**
 * @brief arm concatenation f16 y
 */
void arm_concatenation_f16_y(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_y,
                             uint32_t offset_y);

/**
 * @brief arm concatenation f16 z
 */
void arm_concatenation_f16_z(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             int32_t output_z,
                             uint32_t offset_z);

/**
 * @brief arm concatenation f16 w
 */
void arm_concatenation_f16_w(const float16_t *input,
                             int32_t input_x,
                             int32_t input_y,
                             int32_t input_z,
                             int32_t input_w,
                             float16_t *output,
                             uint32_t offset_w);

/** @} */

/**
 * @addtogroup Pad
 * @{
 */

/**
 * @copydoc arm_pad_f32
 */
arm_cmsis_nn_status arm_pad_f16(const float16_t *input,
                                float16_t *output,
                                float16_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad);

/** @} */

/**
 * @addtogroup NNSupport
 * @{
 */

/**
 * @copydoc arm_batch_norm_f32
 */
arm_cmsis_nn_status arm_batch_norm_f16(const float16_t *input,
                                       float16_t *output,
                                       const float16_t *scale,
                                       const float16_t *bias,
                                       const cmsis_nn_dims *input_dims,
                                       arm_nn_tensor_layout layout);

/**
 * @copydoc arm_reshape_f32
 */
void arm_reshape_f16(const float16_t *input, float16_t *output, uint32_t total_size);

/** @} */

/**
 * @addtogroup FC
 * @{
 */

/**
 * @copydoc arm_batch_matmul_f32
 */
arm_cmsis_nn_status arm_batch_matmul_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params_f16 *bmm_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const float16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const float16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output);

/**
 * @copydoc arm_batch_matmul_f32_get_buffer_size
 */
int32_t arm_batch_matmul_f16_get_buffer_size(const cmsis_nn_bmm_params_f16 *bmm_params,
                                             const cmsis_nn_dims *input_lhs_dims,
                                             const cmsis_nn_dims *input_rhs_dims,
                                             const cmsis_nn_dims *output_dims);

/** @} */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @copydoc arm_transpose_conv_wrapper_f32
 */
arm_cmsis_nn_status arm_transpose_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_context *output_ctx,
                                                   const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output_data,
                                                   arm_nn_tensor_layout layout);

/**
 * @copydoc arm_transpose_conv_nhwc_f32
 */
arm_cmsis_nn_status arm_transpose_conv_nhwc_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_context *output_ctx,
                                                const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output_data);

/**
 * @copydoc arm_transpose_conv_f32
 */
arm_cmsis_nn_status arm_transpose_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output_data,
                                           arm_nn_tensor_layout layout);

/**
 * @copydoc arm_transpose_conv_f32_get_buffer_size
 */
int32_t arm_transpose_conv_f16_get_buffer_size(const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims);

/**
 * @copydoc arm_transpose_conv_f32_get_reverse_conv_buffer_size
 */
int32_t
arm_transpose_conv_f16_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params_f16 *transpose_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims);

/** @} */

/**
 * @addtogroup SVDF
 * @{
 */

/**
 * @copydoc arm_svdf_f32
 */
arm_cmsis_nn_status arm_svdf_f16(const cmsis_nn_context *ctx,
                                 const cmsis_nn_context *input_ctx,
                                 const cmsis_nn_context *output_ctx,
                                 const cmsis_nn_svdf_params_f16 *svdf_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float16_t *input_data,
                                 const cmsis_nn_dims *state_dims,
                                 float16_t *state_data,
                                 const cmsis_nn_dims *weights_feature_dims,
                                 const float16_t *weights_feature_data,
                                 const cmsis_nn_dims *weights_time_dims,
                                 const float16_t *weights_time_data,
                                 const cmsis_nn_dims *bias_dims,
                                 const float16_t *bias_data,
                                 const cmsis_nn_dims *output_dims,
                                 float16_t *output_data);

/** @} */

/**
 * @addtogroup LSTM
 * @{
 */

/**
 * @copydoc arm_lstm_unidirectional_f32
 */
arm_cmsis_nn_status arm_lstm_unidirectional_f16(const float16_t *input,
                                                float16_t *output,
                                                const cmsis_nn_lstm_params_f16 *params,
                                                cmsis_nn_lstm_context_f16 *buffers);

/** @} */

/**
 * @addtogroup Softmax
 * @{
 */

/**
 * @copydoc arm_softmax_f32
 */
arm_cmsis_nn_status arm_softmax_f16(const float16_t *input, int32_t num_rows, int32_t row_size, float16_t *output);

    /** @} */

#endif /* ARM_NN_ENABLE_F16 */

#ifdef __cplusplus
}
#endif

#endif /* ARM_NNFUNCTIONS_FLT_H */
