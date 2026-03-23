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
class PadCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    input_shape: tuple[int, int, int, int]
    pre_pad: tuple[int, int, int, int]
    post_pad: tuple[int, int, int, int]
    pad_value: float
    seed: int
    input_min: float = -4.0
    input_max: float = 4.0


class PadSettingsFlt:
    def __init__(self, case: PadCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input_tensor.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_input(self, array: np.ndarray) -> None:
        save_pregenerated_array(self.input_table_file, array)

    def _load_input(self) -> np.ndarray:
        return load_pregenerated_array(self.input_table_file, self.case.input_shape)

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=self.case.input_shape).astype(np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_input(input_data)
        return input_data

    def _pad_reference(self, input_array: np.ndarray) -> np.ndarray:
        out_shape = tuple(self.case.pre_pad[i] + self.case.input_shape[i] + self.case.post_pad[i] for i in range(4))
        output = np.full(out_shape, self.case.pad_value, dtype=np.float32)
        output[
            self.case.pre_pad[0]:self.case.pre_pad[0] + self.case.input_shape[0],
            self.case.pre_pad[1]:self.case.pre_pad[1] + self.case.input_shape[1],
            self.case.pre_pad[2]:self.case.pre_pad[2] + self.case.input_shape[2],
            self.case.pre_pad[3]:self.case.pre_pad[3] + self.case.input_shape[3],
        ] = input_array
        return output

    def _write_config_header(self, output_shape: tuple[int, int, int, int]) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_N {self.case.input_shape[0]}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_shape[1]}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_shape[2]}\n")
            fh.write(f"#define {prefix}_INPUT_C {self.case.input_shape[3]}\n")
            fh.write(f"#define {prefix}_PRE_PAD_N {self.case.pre_pad[0]}\n")
            fh.write(f"#define {prefix}_PRE_PAD_H {self.case.pre_pad[1]}\n")
            fh.write(f"#define {prefix}_PRE_PAD_W {self.case.pre_pad[2]}\n")
            fh.write(f"#define {prefix}_PRE_PAD_C {self.case.pre_pad[3]}\n")
            fh.write(f"#define {prefix}_POST_PAD_N {self.case.post_pad[0]}\n")
            fh.write(f"#define {prefix}_POST_PAD_H {self.case.post_pad[1]}\n")
            fh.write(f"#define {prefix}_POST_PAD_W {self.case.post_pad[2]}\n")
            fh.write(f"#define {prefix}_POST_PAD_C {self.case.post_pad[3]}\n")
            fh.write(f"#define {prefix}_OUTPUT_N {output_shape[0]}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_shape[1]}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_shape[2]}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {output_shape[3]}\n")
            fh.write(f"#define {prefix}_OUTPUT_SIZE {int(np.prod(output_shape))}\n")
            fh.write(f"#define {prefix}_PAD_VALUE {formatter(self.case.pad_value)}\n")
        format_generated_file(filepath)

    def _write_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{name}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_data = self.get_input_data(regenerate_input)
        output_data = self._pad_reference(input_data)

        input_tensor = torch.tensor(input_data, dtype=self.case.torch_dtype)
        output_tensor = torch.tensor(output_data, dtype=self.case.torch_dtype)
        input_array = input_tensor.cpu().numpy()
        output_array = output_tensor.cpu().numpy()
        if self.case.dtype_name == "f16":
            input_array = input_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_data.shape)
        self._write_array_header("input_tensor", input_array)
        self._write_array_header("output_ref", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[PadCaseFlt]:
    return [
        PadCaseFlt("pad_int8_1_f32", "f32", torch.float32, (1, 2, 2, 2), (0, 0, 1, 1), (0, 0, 2, 2), -3.0, 5001),
        PadCaseFlt("pad_int8_2_f32", "f32", torch.float32, (1, 2, 2, 2), (0, 2, 2, 0), (0, 1, 1, 0), -2.5, 5002),
        PadCaseFlt("pad_basic_f32", "f32", torch.float32, (1, 7, 9, 5), (0, 2, 1, 0), (0, 1, 2, 0), 0.125, 5003),
        PadCaseFlt("pad_int8_1_f16", "f16", torch.float16, (1, 2, 2, 2), (0, 0, 1, 1), (0, 0, 2, 2), -3.0, 5101),
        PadCaseFlt("pad_int8_2_f16", "f16", torch.float16, (1, 2, 2, 2), (0, 2, 2, 0), (0, 1, 1, 0), -2.5, 5102),
        PadCaseFlt("pad_basic_f16", "f16", torch.float16, (1, 7, 9, 5), (0, 2, 1, 0), (0, 1, 2, 0), 0.125, 5103),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float pad unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset or family to generate. Supported values: all, pad_f32_family, pad_f16_family, or a specific dataset name.",
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def _matches(dataset_filter: str, case: PadCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "pad_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "pad_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    matches = [case for case in _build_cases() if _matches(args.dataset, case)]
    if not matches:
        raise RuntimeError(f"No pad dataset matched '{args.dataset}'")

    for case in matches:
        print(f"Generating {case.dataset}...")
        PadSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
