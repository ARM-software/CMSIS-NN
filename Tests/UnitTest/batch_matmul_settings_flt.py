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
class BatchMatmulCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    lhs_batch: int
    lhs_height: int
    lhs_rows: int
    lhs_cols: int
    rhs_batch: int
    rhs_height: int
    rhs_rows: int
    rhs_cols: int
    adj_x: int
    adj_y: int
    activation_min: float
    activation_max: float
    seed: int
    input_min: float = -2.0
    input_max: float = 2.0


class BatchMatmulSettingsFlt:
    def __init__(self, case: BatchMatmulCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.lhs_input_table_file = self.pregenerated_data_dir / "lhs_input.txt"
        self.rhs_input_table_file = self.pregenerated_data_dir / "rhs_input.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, path: Path, values: np.ndarray) -> None:
        save_pregenerated_array(path, values)

    def _load_array(self, path: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(path, shape)

    def get_lhs_input(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.case.lhs_batch, self.case.lhs_height, self.case.lhs_rows, self.case.lhs_cols)
        if self.lhs_input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.lhs_input_table_file}")
            return self._load_array(self.lhs_input_table_file, shape)

        rng = np.random.default_rng(self.case.seed)
        lhs = rng.uniform(self.case.input_min, self.case.input_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.lhs_input_table_file}")
        self._save_array(self.lhs_input_table_file, lhs)
        return lhs

    def get_rhs_input(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.case.rhs_batch, self.case.rhs_height, self.case.rhs_rows, self.case.rhs_cols)
        if self.rhs_input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.rhs_input_table_file}")
            return self._load_array(self.rhs_input_table_file, shape)

        rng = np.random.default_rng(self.case.seed + 101)
        rhs = rng.uniform(self.case.input_min, self.case.input_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.rhs_input_table_file}")
        self._save_array(self.rhs_input_table_file, rhs)
        return rhs

    def _write_config_header(self, output_shape: tuple[int, int, int, int]) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        output_batch, output_height, output_rows, output_cols = output_shape
        dst_size = output_batch * output_height * output_rows * output_cols

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_LHS_BATCH {self.case.lhs_batch}\n")
            fh.write(f"#define {prefix}_LHS_HEIGHT {self.case.lhs_height}\n")
            fh.write(f"#define {prefix}_LHS_ROWS {self.case.lhs_rows}\n")
            fh.write(f"#define {prefix}_LHS_COLS {self.case.lhs_cols}\n")
            fh.write(f"#define {prefix}_RHS_BATCH {self.case.rhs_batch}\n")
            fh.write(f"#define {prefix}_RHS_HEIGHT {self.case.rhs_height}\n")
            fh.write(f"#define {prefix}_RHS_ROWS {self.case.rhs_rows}\n")
            fh.write(f"#define {prefix}_RHS_COLS {self.case.rhs_cols}\n")
            fh.write(f"#define {prefix}_ADJ_X {self.case.adj_x}\n")
            fh.write(f"#define {prefix}_ADJ_Y {self.case.adj_y}\n")
            fh.write(f"#define {prefix}_DST_SIZE {dst_size}\n")
            fh.write(f"#define {prefix}_OUTPUT_BATCH {output_batch}\n")
            fh.write(f"#define {prefix}_OUTPUT_HEIGHT {output_height}\n")
            fh.write(f"#define {prefix}_OUTPUT_ROWS {output_rows}\n")
            fh.write(f"#define {prefix}_OUTPUT_COLS {output_cols}\n")
            fh.write(f"#define {prefix}_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
        format_generated_file(filepath)

    def _write_named_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{name}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def _emulate_batch_matmul_output(self,
                                     lhs_input: np.ndarray,
                                     lhs_transposed: np.ndarray,
                                     rhs_input: np.ndarray,
                                     rhs_transposed: np.ndarray) -> np.ndarray:
        lhs_tensor = lhs_transposed if self.case.adj_x else lhs_input
        rhs_tensor = rhs_input if self.case.adj_y else rhs_transposed

        lhs_rows = self.case.lhs_cols if self.case.adj_x else self.case.lhs_rows
        rhs_rows = self.case.rhs_rows if self.case.adj_y else self.case.rhs_cols
        rhs_cols = self.case.rhs_cols if self.case.adj_y else self.case.rhs_rows

        output_batch = max(self.case.lhs_batch, self.case.rhs_batch)
        output_height = max(self.case.lhs_height, self.case.rhs_height)
        output = np.zeros((output_batch, output_height, lhs_rows, rhs_rows), dtype=np.float32)

        input_lhs = lhs_tensor.reshape(-1)
        input_rhs = rhs_tensor.reshape(-1)

        inner_lhs_diff = 0 if self.case.lhs_height >= self.case.rhs_height else lhs_rows * rhs_cols
        inner_rhs_diff = rhs_rows * rhs_cols if self.case.rhs_height >= self.case.lhs_height else 0
        outer_lhs_diff = (
            inner_lhs_diff if self.case.lhs_batch >= self.case.rhs_batch
            else -((lhs_rows * rhs_cols) - inner_lhs_diff) * self.case.lhs_height
        )
        outer_rhs_diff = (
            (rhs_rows * rhs_cols) - inner_rhs_diff if self.case.rhs_batch >= self.case.lhs_batch
            else -inner_rhs_diff * self.case.rhs_height
        )

        lhs_offset = 0
        rhs_offset = 0
        for i_out_batch in range(output_batch):
            for i_out_height in range(output_height):
                lhs_mat = input_lhs[lhs_offset:]
                rhs_mat = input_rhs[rhs_offset:]
                for i_lhs_rows in range(lhs_rows):
                    for i_rhs_rows in range(rhs_rows):
                        acc = 0.0
                        for k in range(rhs_cols):
                            lhs_v = (
                                lhs_mat[k * lhs_rows + i_lhs_rows]
                                if self.case.adj_x
                                else lhs_mat[i_lhs_rows * rhs_cols + k]
                            )
                            rhs_v = (
                                rhs_mat[k * rhs_rows + i_rhs_rows]
                                if self.case.adj_y
                                else rhs_mat[i_rhs_rows * rhs_cols + k]
                            )
                            acc += lhs_v * rhs_v
                        output[i_out_batch, i_out_height, i_lhs_rows, i_rhs_rows] = np.clip(
                            acc, self.case.activation_min, self.case.activation_max
                        )
                lhs_offset += lhs_rows * rhs_cols
                lhs_offset -= inner_lhs_diff
                rhs_offset += inner_rhs_diff
            lhs_offset += outer_lhs_diff
            rhs_offset += outer_rhs_diff

        return output

    def generate_data(self, regenerate_input: bool = False) -> None:
        lhs_input = self.get_lhs_input(regenerate_input)
        rhs_input = self.get_rhs_input(regenerate_input)

        lhs_tensor = torch.tensor(lhs_input, dtype=torch.float32)
        rhs_tensor = torch.tensor(rhs_input, dtype=torch.float32)
        lhs_transposed = lhs_tensor.transpose(-1, -2).contiguous()
        rhs_transposed = rhs_tensor.transpose(-1, -2).contiguous()

        output = self._emulate_batch_matmul_output(
            lhs_tensor.cpu().numpy(),
            lhs_transposed.cpu().numpy(),
            rhs_tensor.cpu().numpy(),
            rhs_transposed.cpu().numpy(),
        )

        output_shape = tuple(int(dim) for dim in output.shape)

        lhs_input_array = lhs_tensor.to(self.case.torch_dtype).cpu().numpy()
        lhs_transposed_array = lhs_transposed.to(self.case.torch_dtype).cpu().numpy()
        rhs_input_array = rhs_tensor.to(self.case.torch_dtype).cpu().numpy()
        rhs_transposed_array = rhs_transposed.to(self.case.torch_dtype).cpu().numpy()
        output_array = output.astype(np.float32, copy=False)
        if self.case.dtype_name == "f16":
            lhs_input_array = lhs_input_array.astype(np.float16)
            lhs_transposed_array = lhs_transposed_array.astype(np.float16)
            rhs_input_array = rhs_input_array.astype(np.float16)
            rhs_transposed_array = rhs_transposed_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_shape)
        self._write_named_array_header("lhs_input_tensor", lhs_input_array)
        self._write_named_array_header("lhs_transposed_tensor", lhs_transposed_array)
        self._write_named_array_header("rhs_input_tensor", rhs_input_array)
        self._write_named_array_header("rhs_transposed_tensor", rhs_transposed_array)
        self._write_named_array_header("output", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[BatchMatmulCaseFlt]:
    base_cases = [
        ("batch_matmul_1", 2, 2, 8, 5, 2, 2, 5, 7, 0, 0, -10.0, 10.0, 3101),
        ("batch_matmul_2", 2, 2, 8, 5, 1, 1, 7, 5, 0, 1, -10.0, 10.0, 3102),
        ("batch_matmul_3", 1, 1, 5, 8, 2, 2, 5, 7, 1, 0, -10.0, 10.0, 3103),
        ("batch_matmul_4", 2, 1, 5, 8, 1, 2, 7, 5, 1, 1, -10.0, 10.0, 3104),
        ("batch_matmul_5", 1, 2, 8, 5, 2, 1, 7, 5, 0, 1, -1.0, 1.0, 3105),
        # Match the geometry used by the first s8 batch matmul regression case.
        ("batch_matmul_6", 3, 3, 32, 16, 3, 3, 16, 24, 0, 0, -10.0, 10.0, 3106),
        # Match the smallest ET float batch-matmul case:
        # lhs [1, 2, 3], rhs [1, 3, 4], rhs provided transposed to CMSIS.
        ("batch_matmul_et_small", 1, 1, 2, 3, 1, 1, 3, 4, 0, 0, -10.0, 10.0, 3191),
    ]
    cases: list[BatchMatmulCaseFlt] = []
    for base_name, lb, lh, lr, lc, rb, rh, rr, rc, adj_x, adj_y, act_min, act_max, seed in base_cases:
        cases.append(
            BatchMatmulCaseFlt(
                dataset=f"{base_name}_f32",
                dtype_name="f32",
                torch_dtype=torch.float32,
                lhs_batch=lb,
                lhs_height=lh,
                lhs_rows=lr,
                lhs_cols=lc,
                rhs_batch=rb,
                rhs_height=rh,
                rhs_rows=rr,
                rhs_cols=rc,
                adj_x=adj_x,
                adj_y=adj_y,
                activation_min=act_min,
                activation_max=act_max,
                seed=seed,
            )
        )
        cases.append(
            BatchMatmulCaseFlt(
                dataset=f"{base_name}_f16",
                dtype_name="f16",
                torch_dtype=torch.float16,
                lhs_batch=lb,
                lhs_height=lh,
                lhs_rows=lr,
                lhs_cols=lc,
                rhs_batch=rb,
                rhs_height=rh,
                rhs_rows=rr,
                rhs_cols=rc,
                adj_x=adj_x,
                adj_y=adj_y,
                activation_min=act_min,
                activation_max=act_max,
                seed=seed + 100,
            )
        )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float batch matmul unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, batch_matmul_f32_family, "
            "batch_matmul_f16_family, or a specific dataset name such as batch_matmul_3_f16."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def _select_cases(all_cases: list[BatchMatmulCaseFlt], selector: str) -> list[BatchMatmulCaseFlt]:
    if selector == "all":
        return all_cases
    if selector == "batch_matmul_f32_family":
        return [case for case in all_cases if case.dtype_name == "f32"]
    if selector == "batch_matmul_f16_family":
        return [case for case in all_cases if case.dtype_name == "f16"]
    return [case for case in all_cases if case.dataset == selector]


def main() -> None:
    args = parse_args()
    cases = _select_cases(_build_cases(), args.dataset)
    if not cases:
        raise RuntimeError(f"Unknown dataset selector: {args.dataset}")

    for case in cases:
        BatchMatmulSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
