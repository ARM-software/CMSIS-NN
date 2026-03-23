#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License. You may obtain a copy
# of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from float_test_gen_utils import (
    format_generated_file,
    format_float16,
    format_float32,
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


@dataclass(frozen=True)
class DepthwiseConvCaseFlt:
    dataset: str
    dtype_name: str
    layout: str
    use_wrapper: bool
    batches: int
    input_h: int
    input_w: int
    input_channels: int
    channel_multiplier: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    dilation_h: int
    dilation_w: int
    activation_min: float
    activation_max: float
    seed: int
    input_min: float = -1.0
    input_max: float = 1.0
    weight_min: float = -0.75
    weight_max: float = 0.75
    bias_min: float = -0.5
    bias_max: float = 0.5
    use_null_bias: bool = False

    @property
    def output_channels(self) -> int:
        return self.input_channels * self.channel_multiplier


class DepthwiseConvSettingsFlt:
    def __init__(self, case: DepthwiseConvCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.weights_table_file = self.pregenerated_data_dir / "weights.txt"
        self.bias_table_file = self.pregenerated_data_dir / "bias.txt"

        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, filepath: Path, array: np.ndarray) -> None:
        save_pregenerated_array(filepath, array)

    def _load_array(self, filepath: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(filepath, shape)

    def _get_raw_tensors(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_shape = (self.case.batches, self.case.input_channels, self.case.input_h, self.case.input_w)
        weight_shape = (self.case.output_channels, 1, self.case.kernel_h, self.case.kernel_w)
        bias_shape = (self.case.output_channels,)

        if (
            self.input_table_file.exists()
            and self.weights_table_file.exists()
            and self.bias_table_file.exists()
            and not regenerate_input
        ):
            print(f"Loading data from {self.pregenerated_data_dir}")
            return (
                self._load_array(self.input_table_file, input_shape),
                self._load_array(self.weights_table_file, weight_shape),
                self._load_array(self.bias_table_file, bias_shape),
            )

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=input_shape).astype(np.float32)
        weights = rng.uniform(self.case.weight_min, self.case.weight_max, size=weight_shape).astype(np.float32)
        bias = rng.uniform(self.case.bias_min, self.case.bias_max, size=bias_shape).astype(np.float32)
        if self.case.use_null_bias:
            bias.fill(0.0)

        print(f"Saving data to {self.pregenerated_data_dir}")
        self._save_array(self.input_table_file, input_data)
        self._save_array(self.weights_table_file, weights)
        self._save_array(self.bias_table_file, bias)
        return input_data, weights, bias

    def _reorder_input(self, input_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(input_nchw, (0, 2, 3, 1))

    def _reorder_output(self, output_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(output_nchw, (0, 2, 3, 1))

    def _reorder_weights(self, weights_oc1hw: np.ndarray) -> np.ndarray:
        weights_ock = np.squeeze(weights_oc1hw, axis=1)
        return np.transpose(weights_ock, (1, 2, 0))

    def _compute_reference(self, input_nchw: np.ndarray, weights_oc1hw: np.ndarray, bias: np.ndarray) -> np.ndarray:
        input_tensor = torch.tensor(input_nchw, dtype=torch.float32)
        weight_tensor = torch.tensor(weights_oc1hw, dtype=torch.float32)
        bias_tensor = torch.tensor(bias, dtype=torch.float32)
        output = F.conv2d(
            input_tensor,
            weight_tensor,
            bias_tensor,
            stride=(self.case.stride_h, self.case.stride_w),
            padding=(self.case.padding_h, self.case.padding_w),
            dilation=(self.case.dilation_h, self.case.dilation_w),
            groups=self.case.input_channels,
        )
        output = torch.clamp(output, min=self.case.activation_min, max=self.case.activation_max)
        return output.cpu().numpy().astype(np.float32)

    def _write_config_header(self, output_h: int, output_w: int) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        layout_macro = "ARM_NN_LAYOUT_NHWC"
        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.batches}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_IN_CH {self.case.input_channels}\n")
            fh.write(f"#define {prefix}_CH_MULT {self.case.channel_multiplier}\n")
            fh.write(f"#define {prefix}_FILTER_H {self.case.kernel_h}\n")
            fh.write(f"#define {prefix}_FILTER_W {self.case.kernel_w}\n")
            fh.write(f"#define {prefix}_OUT_CH {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_STRIDE_H {self.case.stride_h}\n")
            fh.write(f"#define {prefix}_STRIDE_W {self.case.stride_w}\n")
            fh.write(f"#define {prefix}_PADDING_H {self.case.padding_h}\n")
            fh.write(f"#define {prefix}_PADDING_W {self.case.padding_w}\n")
            fh.write(f"#define {prefix}_DILATION_H {self.case.dilation_h}\n")
            fh.write(f"#define {prefix}_DILATION_W {self.case.dilation_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_h}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.batches * output_h * output_w * self.case.output_channels}\n")
            fh.write(f"#define {prefix}_LAYOUT {layout_macro}\n")
            fh.write(f"#define {prefix}_USE_WRAPPER {1 if self.case.use_wrapper else 0}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
            fh.write(f"#define {prefix}_USE_NULL_BIAS {1 if self.case.use_null_bias else 0}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = suffix
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        symbol_suffix = Path(suffix).stem
        write_float_array_header(
            filepath,
            self.script_name,
            f"{self.case.dataset}_{symbol_suffix}",
            values,
            self.case.dtype_name,
        )

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_nchw, weights_oc1hw, bias = self._get_raw_tensors(regenerate_input)
        output_nchw = self._compute_reference(input_nchw, weights_oc1hw, bias)
        output_h = int(output_nchw.shape[2])
        output_w = int(output_nchw.shape[3])

        input_export = self._reorder_input(input_nchw)
        weights_export = self._reorder_weights(weights_oc1hw)
        output_export = self._reorder_output(output_nchw)

        if self.case.dtype_name == "f16":
            input_export = input_export.astype(np.float16)
            weights_export = weights_export.astype(np.float16)
            bias = bias.astype(np.float16)
            output_export = output_export.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_h, output_w)
        self._write_array_header("input_data.h", input_export)
        self._write_array_header("weights_data.h", weights_export)
        self._write_array_header("biases_data.h", bias)
        self._write_array_header("output_ref_data.h", output_export)
        self._write_wrapper_header()


def _build_cases() -> list[DepthwiseConvCaseFlt]:
    base_cases = [
        DepthwiseConvCaseFlt(
            dataset="depthwise_basic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=9,
            input_w=11,
            input_channels=8,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2401,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_basic_smallc_nhwc_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=9,
            input_w=11,
            input_channels=6,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2402,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_k3_1d_opt_batch2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=1,
            input_w=21,
            input_channels=8,
            channel_multiplier=1,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2403,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_k3_1d_opt_nhwc_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=1,
            input_w=21,
            input_channels=8,
            channel_multiplier=1,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2404,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_2x5_opt_batch2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=2,
            input_w=21,
            input_channels=8,
            channel_multiplier=1,
            kernel_h=2,
            kernel_w=5,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2405,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_2x5_opt_nhwc_chmult16_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=2,
            input_w=21,
            input_channels=1,
            channel_multiplier=16,
            kernel_h=2,
            kernel_w=5,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2406,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_ic1_to_conv_nhwc_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=9,
            input_w=11,
            input_channels=1,
            channel_multiplier=16,
            kernel_h=4,
            kernel_w=4,
            stride_h=2,
            stride_w=1,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2410,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_kernel_2x2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=6,
            input_w=5,
            input_channels=4,
            channel_multiplier=1,
            kernel_h=2,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2407,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_kernel_3x3_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=5,
            input_w=4,
            input_channels=5,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            padding_h=1,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2408,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_kernel_3x3_null_bias_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=5,
            input_w=4,
            input_channels=5,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            padding_h=1,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2409,
            use_null_bias=True,
        ),
        # Match representative legacy s8 depthwise regression shapes for direct cycle comparisons.
        DepthwiseConvCaseFlt(
            dataset="depthwise_match_basic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=8,
            input_w=5,
            input_channels=1,
            channel_multiplier=1,
            kernel_h=4,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2411,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_match_sub_block_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=5,
            input_w=7,
            input_channels=9,
            channel_multiplier=1,
            kernel_h=2,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2412,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_match_dilation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=7,
            input_w=7,
            input_channels=1,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=2,
            padding_w=2,
            dilation_h=2,
            dilation_w=2,
            activation_min=-70.0,
            activation_max=127.0,
            seed=2413,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_match_out_activation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=5,
            input_w=6,
            input_channels=3,
            channel_multiplier=1,
            kernel_h=4,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-45.0,
            activation_max=103.0,
            seed=2414,
        ),
        DepthwiseConvCaseFlt(
            dataset="depthwise_match_stride2pad1_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=7,
            input_w=7,
            input_channels=1,
            channel_multiplier=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2415,
        ),
    ]

    f16_cases = [
        DepthwiseConvCaseFlt(**{**case.__dict__, "dataset": case.dataset.replace("_f32", "_f16"), "dtype_name": "f16"})
        for case in base_cases
    ]
    return base_cases + f16_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float depthwise-convolution unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, depthwise_conv_f32_family, "
            "depthwise_conv_f16_family, or a specific dataset name such as depthwise_basic_smallc_nhwc_f16."
        ),
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate the pregenerated raw input/weight/bias tensors instead of reusing them.",
    )
    return parser.parse_args()


def _select_cases(all_cases: list[DepthwiseConvCaseFlt], selector: str) -> list[DepthwiseConvCaseFlt]:
    if selector == "all":
        return all_cases
    if selector == "depthwise_conv_f32_family":
        return [case for case in all_cases if case.dtype_name == "f32"]
    if selector == "depthwise_conv_f16_family":
        return [case for case in all_cases if case.dtype_name == "f16"]
    return [case for case in all_cases if case.dataset == selector]


def main() -> None:
    args = parse_args()
    selected = _select_cases(_build_cases(), args.dataset)
    if not selected:
        raise SystemExit(f"Unknown dataset selector: {args.dataset}")

    for case in selected:
        print(f"Generating {case.dataset}...")
        DepthwiseConvSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
