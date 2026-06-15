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
class SvdfCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    input_batches: int
    input_size: int
    unit_count: int
    rank: int
    memory_size: int
    sequence_steps: int
    seed: int
    input_act_min: float
    input_act_max: float
    output_act_min: float
    output_act_max: float
    input_min: float = -1.5
    input_max: float = 1.5
    weight_min: float = -0.5
    weight_max: float = 0.5
    bias_min: float = -0.25
    bias_max: float = 0.25


class SvdfSettingsFlt:
    def __init__(self, case: SvdfCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_sequence_file = self.pregenerated_data_dir / "input_sequence.txt"
        self.weights_feature_file = self.pregenerated_data_dir / "weights_feature.txt"
        self.weights_time_file = self.pregenerated_data_dir / "weights_time.txt"
        self.bias_file = self.pregenerated_data_dir / "bias.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    @property
    def feature_batches(self) -> int:
        return self.case.unit_count * self.case.rank

    def _save_array(self, path: Path, values: np.ndarray) -> None:
        save_pregenerated_array(path, values)

    def _load_array(self, path: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(path, shape)

    def get_input_sequence(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.case.sequence_steps, self.case.input_batches, self.case.input_size)
        if self.input_sequence_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_sequence_file}")
            return self._load_array(self.input_sequence_file, shape)

        rng = np.random.default_rng(self.case.seed + 1)
        values = rng.uniform(self.case.input_min, self.case.input_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.input_sequence_file}")
        self._save_array(self.input_sequence_file, values)
        return values

    def get_weights_feature(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.feature_batches, self.case.input_size)
        if self.weights_feature_file.exists() and not regenerate_input:
            print(f"Loading data from {self.weights_feature_file}")
            return self._load_array(self.weights_feature_file, shape)

        rng = np.random.default_rng(self.case.seed + 2)
        values = rng.uniform(self.case.weight_min, self.case.weight_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.weights_feature_file}")
        self._save_array(self.weights_feature_file, values)
        return values

    def get_weights_time(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.feature_batches, self.case.memory_size)
        if self.weights_time_file.exists() and not regenerate_input:
            print(f"Loading data from {self.weights_time_file}")
            return self._load_array(self.weights_time_file, shape)

        rng = np.random.default_rng(self.case.seed + 3)
        values = rng.uniform(self.case.weight_min, self.case.weight_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.weights_time_file}")
        self._save_array(self.weights_time_file, values)
        return values

    def get_bias(self, regenerate_input: bool) -> np.ndarray:
        shape = (self.case.unit_count,)
        if self.bias_file.exists() and not regenerate_input:
            print(f"Loading data from {self.bias_file}")
            return self._load_array(self.bias_file, shape)

        rng = np.random.default_rng(self.case.seed + 4)
        values = rng.uniform(self.case.bias_min, self.case.bias_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.bias_file}")
        self._save_array(self.bias_file, values)
        return values

    def _svdf_reference(self, input_sequence: np.ndarray, weights_feature: np.ndarray, weights_time: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros((self.case.input_batches, self.feature_batches, self.case.memory_size), dtype=np.float32)
        output = np.zeros((self.case.input_batches, self.case.unit_count), dtype=np.float32)

        for step in range(self.case.sequence_steps):
            state[:, :, :-1] = state[:, :, 1:]
            projected = input_sequence[step] @ weights_feature.T
            projected = np.clip(projected, self.case.input_act_min, self.case.input_act_max)
            state[:, :, -1] = projected

            out_a = np.sum(state * weights_time[None, :, :], axis=2)
            out_b = out_a.reshape(self.case.input_batches, self.case.unit_count, self.case.rank).sum(axis=2) + bias[None, :]
            output = np.clip(out_b, self.case.output_act_min, self.case.output_act_max)

        return state, output

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.input_batches}\n")
            fh.write(f"#define {prefix}_INPUT_SIZE {self.case.input_size}\n")
            fh.write(f"#define {prefix}_UNIT_COUNT {self.case.unit_count}\n")
            fh.write(f"#define {prefix}_RANK {self.case.rank}\n")
            fh.write(f"#define {prefix}_FEATURE_BATCHES {self.feature_batches}\n")
            fh.write(f"#define {prefix}_TIME_BATCHES {self.case.memory_size}\n")
            fh.write(f"#define {prefix}_SEQUENCE_STEPS {self.case.sequence_steps}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.input_batches * self.case.unit_count}\n")
            fh.write(f"#define {prefix}_INPUT_ACTIVATION_MIN {formatter(self.case.input_act_min)}\n")
            fh.write(f"#define {prefix}_INPUT_ACTIVATION_MAX {formatter(self.case.input_act_max)}\n")
            fh.write(f"#define {prefix}_OUTPUT_ACTIVATION_MIN {formatter(self.case.output_act_min)}\n")
            fh.write(f"#define {prefix}_OUTPUT_ACTIVATION_MAX {formatter(self.case.output_act_max)}\n")
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
        input_sequence = self.get_input_sequence(regenerate_input)
        weights_feature = self.get_weights_feature(regenerate_input)
        weights_time = self.get_weights_time(regenerate_input)
        bias = self.get_bias(regenerate_input)
        state_ref, output_ref = self._svdf_reference(input_sequence, weights_feature, weights_time, bias)

        arrays = {
            "input_sequence": torch.tensor(input_sequence, dtype=self.case.torch_dtype).cpu().numpy(),
            "weights_feature": torch.tensor(weights_feature, dtype=self.case.torch_dtype).cpu().numpy(),
            "weights_time": torch.tensor(weights_time, dtype=self.case.torch_dtype).cpu().numpy(),
            "bias": torch.tensor(bias, dtype=self.case.torch_dtype).cpu().numpy(),
            "initial_state": np.zeros_like(state_ref, dtype=np.float16 if self.case.dtype_name == "f16" else np.float32),
            "output_ref": torch.tensor(output_ref, dtype=self.case.torch_dtype).cpu().numpy(),
        }
        if self.case.dtype_name == "f16":
            arrays = {name: value.astype(np.float16) for name, value in arrays.items()}

        self.generated_header_files = []
        self._write_config_header()
        for name, values in arrays.items():
            self._write_array_header(name, values)
        self._write_wrapper_header()


def _build_cases() -> list[SvdfCaseFlt]:
    return [
        SvdfCaseFlt("svdf_small_f32", "f32", torch.float32, 1, 16, 6, 2, 4, 3, 7001, -1.0, 1.0, -100.0, 100.0),
        SvdfCaseFlt("svdf_batch2_f32", "f32", torch.float32, 2, 20, 8, 2, 3, 4, 7002, -1.0, 1.0, -100.0, 100.0),
        # Match legacy s8 svdf_int8 geometry.
        SvdfCaseFlt("svdf_match_1_f32", "f32", torch.float32, 1, 20, 12, 1, 2, 3, 7003, -1.0, 1.0, -100.0, 100.0),
        # Match legacy s8 svdf_int8_2 geometry.
        SvdfCaseFlt("svdf_match_2_f32", "f32", torch.float32, 2, 40, 13, 2, 3, 4, 7004, -1.0, 1.0, -100.0, 100.0),
        SvdfCaseFlt("svdf_small_f16", "f16", torch.float16, 1, 16, 6, 2, 4, 3, 7101, -1.0, 1.0, -100.0, 100.0),
        SvdfCaseFlt("svdf_batch2_f16", "f16", torch.float16, 2, 20, 8, 2, 3, 4, 7102, -1.0, 1.0, -100.0, 100.0),
        SvdfCaseFlt("svdf_match_1_f16", "f16", torch.float16, 1, 20, 12, 1, 2, 3, 7103, -1.0, 1.0, -100.0, 100.0),
        SvdfCaseFlt("svdf_match_2_f16", "f16", torch.float16, 2, 40, 13, 2, 3, 4, 7104, -1.0, 1.0, -100.0, 100.0),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float SVDF unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset or family to generate. Supported values: all, svdf_f32_family, svdf_f16_family, or a specific dataset name.",
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def _matches(dataset_filter: str, case: SvdfCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "svdf_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "svdf_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    matches = [case for case in _build_cases() if _matches(args.dataset, case)]
    if not matches:
        raise RuntimeError(f"No SVDF dataset matched '{args.dataset}'")

    for case in matches:
        print(f"Generating {case.dataset}...")
        SvdfSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
