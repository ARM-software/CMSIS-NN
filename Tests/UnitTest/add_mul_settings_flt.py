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
class AddMulCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    op_name: str
    block_size: int
    activation_min: float
    activation_max: float
    seed: int
    input_min: float
    input_max: float


class AddMulSettingsFlt:
    def __init__(self, case: AddMulCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input1_table_file = self.pregenerated_data_dir / "input1.txt"
        self.input2_table_file = self.pregenerated_data_dir / "input2.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_inputs(self, input1: np.ndarray, input2: np.ndarray) -> None:
        save_pregenerated_array(self.input1_table_file, input1)
        save_pregenerated_array(self.input2_table_file, input2)

    def _load_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        input1 = load_pregenerated_array(self.input1_table_file, (self.case.block_size,))
        input2 = load_pregenerated_array(self.input2_table_file, (self.case.block_size,))
        return input1, input2

    def get_input_data(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.input1_table_file.exists() and self.input2_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input1_table_file} and {self.input2_table_file}")
            return self._load_inputs()

        rng = np.random.default_rng(self.case.seed)
        input1 = rng.uniform(self.case.input_min, self.case.input_max, size=self.case.block_size).astype(np.float32)
        input2 = rng.uniform(self.case.input_min, self.case.input_max, size=self.case.block_size).astype(np.float32)
        print(f"Saving data to {self.input1_table_file} and {self.input2_table_file}")
        self._save_inputs(input1, input2)
        return input1, input2

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_DST_SIZE {self.case.block_size}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
            fh.write(f"#define {prefix}_INPUT1_OFFSET 0\n")
            fh.write(f"#define {prefix}_INPUT2_OFFSET 0\n")
            fh.write(f"#define {prefix}_OUTPUT_MULT 0\n")
            fh.write(f"#define {prefix}_OUTPUT_SHIFT 0\n")
            fh.write(f"#define {prefix}_OUTPUT_OFFSET 0\n")
            if self.case.op_name == "add":
                fh.write(f"#define {prefix}_LEFT_SHIFT 0\n")
                fh.write(f"#define {prefix}_INPUT1_SHIFT 0\n")
                fh.write(f"#define {prefix}_INPUT2_SHIFT 0\n")
                fh.write(f"#define {prefix}_INPUT1_MULT 0\n")
                fh.write(f"#define {prefix}_INPUT2_MULT 0\n")
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
        input1, input2 = self.get_input_data(regenerate_input)
        input1_tensor = torch.tensor(input1, dtype=torch.float32)
        input2_tensor = torch.tensor(input2, dtype=torch.float32)
        if self.case.op_name == "add":
            output_tensor = input1_tensor + input2_tensor
        elif self.case.op_name == "mul":
            output_tensor = input1_tensor * input2_tensor
        else:
            raise RuntimeError("Unsupported elementwise op")

        output_tensor = torch.clamp(output_tensor, min=self.case.activation_min, max=self.case.activation_max)
        input1_array = input1_tensor.to(self.case.torch_dtype).cpu().numpy()
        input2_array = input2_tensor.to(self.case.torch_dtype).cpu().numpy()
        output_array = output_tensor.to(self.case.torch_dtype).cpu().numpy()
        if self.case.dtype_name == "f16":
            input1_array = input1_array.astype(np.float16)
            input2_array = input2_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header()
        self._write_array_header("input1", input1_array)
        self._write_array_header("input2", input2_array)
        self._write_array_header("output_ref", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[AddMulCaseFlt]:
    return [
        AddMulCaseFlt(
            dataset="add_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="add",
            block_size=128,
            activation_min=-10.0,
            activation_max=10.0,
            seed=2001,
            input_min=-4.0,
            input_max=4.0,
        ),
        AddMulCaseFlt(
            dataset="add_f32_spill",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="add",
            block_size=105,
            activation_min=-2.0,
            activation_max=3.0,
            seed=2002,
            input_min=-4.0,
            input_max=4.0,
        ),
        AddMulCaseFlt(
            dataset="add_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="add",
            block_size=128,
            activation_min=-10.0,
            activation_max=10.0,
            seed=2101,
            input_min=-4.0,
            input_max=4.0,
        ),
        AddMulCaseFlt(
            dataset="add_f16_spill",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="add",
            block_size=105,
            activation_min=-2.0,
            activation_max=3.0,
            seed=2102,
            input_min=-4.0,
            input_max=4.0,
        ),
        AddMulCaseFlt(
            dataset="mul_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="mul",
            block_size=160,
            activation_min=-10.0,
            activation_max=10.0,
            seed=2201,
            input_min=-2.5,
            input_max=2.5,
        ),
        AddMulCaseFlt(
            dataset="mul_f32_spill",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="mul",
            block_size=245,
            activation_min=-1.5,
            activation_max=1.0,
            seed=2202,
            input_min=-2.5,
            input_max=2.5,
        ),
        AddMulCaseFlt(
            dataset="mul_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="mul",
            block_size=160,
            activation_min=-10.0,
            activation_max=10.0,
            seed=2301,
            input_min=-2.5,
            input_max=2.5,
        ),
        AddMulCaseFlt(
            dataset="mul_f16_spill",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="mul",
            block_size=245,
            activation_min=-1.5,
            activation_max=1.0,
            seed=2302,
            input_min=-2.5,
            input_max=2.5,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float add/mul unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, add_f32_family, add_f16_family, "
            "mul_f32_family, mul_f16_family, or a specific dataset name such as add_f32_spill."
        ),
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate pregenerated input samples instead of reusing input files.",
    )
    return parser.parse_args()


def _matches_selector(case: AddMulCaseFlt, selector: str) -> bool:
    if selector == "all":
        return True
    if selector == case.dataset:
        return True
    if selector == "add_f32_family":
        return case.dataset.startswith("add_f32")
    if selector == "add_f16_family":
        return case.dataset.startswith("add_f16")
    if selector == "mul_f32_family":
        return case.dataset.startswith("mul_f32")
    if selector == "mul_f16_family":
        return case.dataset.startswith("mul_f16")
    return False


def main() -> None:
    args = parse_args()
    cases = [case for case in _build_cases() if _matches_selector(case, args.dataset)]
    if not cases:
        raise SystemExit(f"Unknown dataset selector: {args.dataset}")

    for case in cases:
        print(f"Generating {case.dataset}...")
        AddMulSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
