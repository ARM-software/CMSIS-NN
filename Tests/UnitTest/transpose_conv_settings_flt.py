#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
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


LAYOUT_VALUE = {"nhwc": "ARM_NN_LAYOUT_NHWC"}


@dataclass(frozen=True)
class TransposeConvCaseFlt:
    dataset: str
    dtype_name: str
    layout: str
    use_wrapper: bool
    batches: int
    input_h: int
    input_w: int
    input_channels: int
    output_channels: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    output_padding_h: int
    output_padding_w: int
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


class TransposeConvSettingsFlt:
    def __init__(self, case: TransposeConvCaseFlt):
        self.case = case
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(case.dataset)
        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.weights_table_file = self.pregenerated_data_dir / "weights.txt"
        self.bias_table_file = self.pregenerated_data_dir / "bias.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, filepath: Path, values: np.ndarray) -> None:
        save_pregenerated_array(filepath, values)

    def _load_array(self, filepath: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(filepath, shape)

    def _get_raw_tensors(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_shape = (self.case.batches, self.case.input_channels, self.case.input_h, self.case.input_w)
        weight_shape = (self.case.input_channels, self.case.output_channels, self.case.kernel_h, self.case.kernel_w)
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
        print(f"Saving data to {self.pregenerated_data_dir}")
        self._save_array(self.input_table_file, input_data)
        self._save_array(self.weights_table_file, weights)
        self._save_array(self.bias_table_file, bias)
        return input_data, weights, bias

    def _reorder_input(self, input_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(input_nchw, (0, 2, 3, 1))

    def _reorder_output(self, output_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(output_nchw, (0, 2, 3, 1))

    def _reorder_weights(self, weights_iohw: np.ndarray) -> np.ndarray:
        return np.transpose(weights_iohw, (1, 2, 3, 0))

    def _compute_reference(self, input_nchw: np.ndarray, weights_iohw: np.ndarray, bias: np.ndarray) -> np.ndarray:
        output = F.conv_transpose2d(
            torch.tensor(input_nchw, dtype=torch.float32),
            torch.tensor(weights_iohw, dtype=torch.float32),
            torch.tensor(bias, dtype=torch.float32),
            stride=(self.case.stride_h, self.case.stride_w),
            padding=(self.case.padding_h, self.case.padding_w),
            output_padding=(self.case.output_padding_h, self.case.output_padding_w),
            dilation=(self.case.dilation_h, self.case.dilation_w),
        )
        return torch.clamp(output, min=self.case.activation_min, max=self.case.activation_max).cpu().numpy().astype(np.float32)

    def _write_config_header(self, output_h: int, output_w: int) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.batches}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_IN_CH {self.case.input_channels}\n")
            fh.write(f"#define {prefix}_FILTER_H {self.case.kernel_h}\n")
            fh.write(f"#define {prefix}_FILTER_W {self.case.kernel_w}\n")
            fh.write(f"#define {prefix}_OUT_CH {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_STRIDE_H {self.case.stride_h}\n")
            fh.write(f"#define {prefix}_STRIDE_W {self.case.stride_w}\n")
            fh.write(f"#define {prefix}_PADDING_H {self.case.padding_h}\n")
            fh.write(f"#define {prefix}_PADDING_W {self.case.padding_w}\n")
            # Keep transpose-convolution padding offsets at zero to match the
            # existing manifest/export convention. PyTorch output_padding is
            # already reflected in the exported output shape and reference.
            fh.write(f"#define {prefix}_PADDING_OFFSET_H 0\n")
            fh.write(f"#define {prefix}_PADDING_OFFSET_W 0\n")
            fh.write(f"#define {prefix}_DILATION_H {self.case.dilation_h}\n")
            fh.write(f"#define {prefix}_DILATION_W {self.case.dilation_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_h}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.batches * output_h * output_w * self.case.output_channels}\n")
            fh.write(f"#define {prefix}_LAYOUT {LAYOUT_VALUE[self.case.layout]}\n")
            fh.write(f"#define {prefix}_USE_WRAPPER {1 if self.case.use_wrapper else 0}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = f"{suffix}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{suffix}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        write_wrapper_header(self.headers_dir / "test_data.h", self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_nchw, weights_iohw, bias = self._get_raw_tensors(regenerate_input)
        output_nchw = self._compute_reference(input_nchw, weights_iohw, bias)
        output_h = output_nchw.shape[2]
        output_w = output_nchw.shape[3]

        input_export = self._reorder_input(input_nchw)
        weights_export = self._reorder_weights(weights_iohw)
        output_export = self._reorder_output(output_nchw)
        bias_export = bias
        if self.case.dtype_name == "f16":
            input_export = input_export.astype(np.float16)
            weights_export = weights_export.astype(np.float16)
            bias_export = bias_export.astype(np.float16)
            output_export = output_export.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_h, output_w)
        self._write_array_header("input", input_export)
        self._write_array_header("weights", weights_export)
        self._write_array_header("biases", bias_export)
        self._write_array_header("output_ref", output_export)
        self._write_wrapper_header()


def build_cases() -> list[TransposeConvCaseFlt]:
    base_cases = [
        ("transpose_conv_basic", "nhwc", False, 8301),
        ("transpose_conv_basic_nhwc", "nhwc", True, 8302),
    ]
    cases: list[TransposeConvCaseFlt] = []
    for dtype_name, seed_offset in (("f32", 0), ("f16", 100)):
        for name, layout, use_wrapper, seed in base_cases:
            cases.append(
                TransposeConvCaseFlt(
                    dataset=f"{name}_{dtype_name}",
                    dtype_name=dtype_name,
                    layout=layout,
                    use_wrapper=use_wrapper,
                    batches=1,
                    input_h=5,
                    input_w=6,
                    input_channels=4,
                    output_channels=6,
                    kernel_h=3,
                    kernel_w=3,
                    stride_h=2,
                    stride_w=2,
                    padding_h=1,
                    padding_w=1,
                    output_padding_h=1,
                    output_padding_w=1,
                    dilation_h=1,
                    dilation_w=1,
                    activation_min=-1000.0,
                    activation_max=1000.0,
                    seed=seed + seed_offset,
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float transpose-convolution unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, transpose_conv_f32_family, "
            "transpose_conv_f16_family, or a specific dataset name."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def matches(dataset_filter: str, case: TransposeConvCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "transpose_conv_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "transpose_conv_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    selected = [case for case in build_cases() if matches(args.dataset, case)]
    if not selected:
        raise RuntimeError(f"No transpose_conv dataset matched '{args.dataset}'")
    for case in selected:
        print(f"Generating {case.dataset}...")
        TransposeConvSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
