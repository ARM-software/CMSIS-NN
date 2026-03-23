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
import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FloatTestFamily:
    name: str
    generator_script: str | None
    selector_by_dtype: dict[str, list[str]]
    cmsis_project: str

    @property
    def host_target_prefix(self) -> str:
        return self.cmsis_project.removesuffix("_flt")

    def host_target(self, dtype_name: str) -> str:
        return f"{self.host_target_prefix}_{dtype_name}"

    def cmsis_context(self, dtype_name: str) -> str:
        return f"{self.cmsis_project}.{dtype_name.upper()}+Corstone-300-FVP"


@dataclass(frozen=True)
class StepResult:
    stage: str
    family: str
    dtype: str
    toolchain: str
    status: str
    detail: str = ""


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_TEST_ROOT = Path(__file__).resolve().parent
CMSIS_UNIT_TEST_ROOT = UNIT_TEST_ROOT / "cmsis"
CMSIS_SOLUTION = CMSIS_UNIT_TEST_ROOT / "cmsis_nn_unit_tests_flt.csolution.yml"
CMSIS_OUTPUT_ROOT = os.environ.get("CMSIS_NN_CBUILD_OUTPUT_ROOT", "").strip()

FAMILY_CONFIGS: dict[str, FloatTestFamily] = {
    "activation": FloatTestFamily(
        name="activation",
        generator_script="activation_settings_flt.py",
        selector_by_dtype={"f32": ["activation_f32"], "f16": ["activation_f16"]},
        cmsis_project="test_arm_activation_flt",
    ),
    "avg_pool": FloatTestFamily(
        name="avg_pool",
        generator_script="pooling_settings_flt.py",
        selector_by_dtype={"f32": ["avgpool_f32"], "f16": ["avgpool_f16"]},
        cmsis_project="test_arm_avg_pool_flt",
    ),
    "batch_matmul": FloatTestFamily(
        name="batch_matmul",
        generator_script="batch_matmul_settings_flt.py",
        selector_by_dtype={"f32": ["batch_matmul_f32_family"], "f16": ["batch_matmul_f16_family"]},
        cmsis_project="test_arm_batch_matmul_flt",
    ),
    "batch_norm": FloatTestFamily(
        name="batch_norm",
        generator_script="batch_norm_settings_flt.py",
        selector_by_dtype={"f32": ["batch_norm_f32_family"], "f16": ["batch_norm_f16_family"]},
        cmsis_project="test_arm_batch_norm_flt",
    ),
    "concatenation": FloatTestFamily(
        name="concatenation",
        generator_script="concatenation_settings_flt.py",
        selector_by_dtype={"f32": ["concatenation_f32_family"], "f16": ["concatenation_f16_family"]},
        cmsis_project="test_arm_concatenation_flt",
    ),
    "convolve": FloatTestFamily(
        name="convolve",
        generator_script="conv_settings_flt.py",
        selector_by_dtype={"f32": ["conv_f32_family"], "f16": ["conv_f16_family"]},
        cmsis_project="test_arm_convolve_flt",
    ),
    "depthwise_conv": FloatTestFamily(
        name="depthwise_conv",
        generator_script="depthwise_conv_settings_flt.py",
        selector_by_dtype={"f32": ["depthwise_conv_f32_family"], "f16": ["depthwise_conv_f16_family"]},
        cmsis_project="test_arm_depthwise_conv_flt",
    ),
    "ds_cnn_s_body": FloatTestFamily(
        name="ds_cnn_s_body",
        generator_script="ds_cnn_s_body_settings_flt.py",
        selector_by_dtype={"f32": ["ds_cnn_s_body_f32_family"], "f16": ["ds_cnn_s_body_f16_family"]},
        cmsis_project="test_arm_ds_cnn_s_body_flt",
    ),
    "elementwise_add": FloatTestFamily(
        name="elementwise_add",
        generator_script="add_mul_settings_flt.py",
        selector_by_dtype={"f32": ["add_f32_family"], "f16": ["add_f16_family"]},
        cmsis_project="test_arm_elementwise_add_flt",
    ),
    "elementwise_mul": FloatTestFamily(
        name="elementwise_mul",
        generator_script="add_mul_settings_flt.py",
        selector_by_dtype={"f32": ["mul_f32_family"], "f16": ["mul_f16_family"]},
        cmsis_project="test_arm_elementwise_mul_flt",
    ),
    "fully_connected": FloatTestFamily(
        name="fully_connected",
        generator_script="fully_connected_settings_flt.py",
        selector_by_dtype={"f32": ["fully_connected_f32_family"], "f16": ["fully_connected_f16_family"]},
        cmsis_project="test_arm_fully_connected_flt",
    ),
    "lstm": FloatTestFamily(
        name="lstm",
        generator_script="lstm_settings_flt.py",
        selector_by_dtype={"f32": ["lstm_f32_family"], "f16": ["lstm_f16_family"]},
        cmsis_project="test_arm_lstm_unidirectional_flt",
    ),
    "max_pool": FloatTestFamily(
        name="max_pool",
        generator_script="pooling_settings_flt.py",
        selector_by_dtype={"f32": ["maxpool_f32"], "f16": ["maxpool_f16"]},
        cmsis_project="test_arm_max_pool_flt",
    ),
    "maximum_minimum": FloatTestFamily(
        name="maximum_minimum",
        generator_script="minmax_settings_flt.py",
        selector_by_dtype={
            "f32": ["maximum_f32_family", "minimum_f32_family"],
            "f16": ["maximum_f16_family", "minimum_f16_family"],
        },
        cmsis_project="test_arm_maximum_minimum_flt",
    ),
    "pad": FloatTestFamily(
        name="pad",
        generator_script="pad_settings_flt.py",
        selector_by_dtype={"f32": ["pad_f32_family"], "f16": ["pad_f16_family"]},
        cmsis_project="test_arm_pad_flt",
    ),
    "reshape": FloatTestFamily(
        name="reshape",
        generator_script=None,
        selector_by_dtype={"f32": [], "f16": []},
        cmsis_project="test_arm_reshape_flt",
    ),
    "softmax": FloatTestFamily(
        name="softmax",
        generator_script="softmax_settings_flt.py",
        selector_by_dtype={"f32": ["softmax_f32"], "f16": ["softmax_f16"]},
        cmsis_project="test_arm_softmax_flt",
    ),
    "svdf": FloatTestFamily(
        name="svdf",
        generator_script="svdf_settings_flt.py",
        selector_by_dtype={"f32": ["svdf_f32_family"], "f16": ["svdf_f16_family"]},
        cmsis_project="test_arm_svdf_flt",
    ),
    "transpose": FloatTestFamily(
        name="transpose",
        generator_script="transpose_settings_flt.py",
        selector_by_dtype={"f32": ["transpose_f32_family"], "f16": ["transpose_f16_family"]},
        cmsis_project="test_arm_transpose_flt",
    ),
    "transpose_conv": FloatTestFamily(
        name="transpose_conv",
        generator_script="transpose_conv_settings_flt.py",
        selector_by_dtype={"f32": ["transpose_conv_f32_family"], "f16": ["transpose_conv_f16_family"]},
        cmsis_project="test_arm_transpose_conv_flt",
    ),
}

FAMILY_ALIASES = {
    "bn": "batch_norm",
    "batchnorm": "batch_norm",
    "concat": "concatenation",
    "conv": "convolve",
    "depthwise": "depthwise_conv",
    "ds_cnn_s": "ds_cnn_s_body",
    "dwconv": "depthwise_conv",
    "fc": "fully_connected",
    "bmm": "batch_matmul",
    "lstm_unidirectional": "lstm",
    "minmax": "maximum_minimum",
    "maximum": "maximum_minimum",
    "minimum": "maximum_minimum",
    "transposeconv": "transpose_conv",
    "tconv": "transpose_conv",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    default_cmsis_path = os.environ.get("CMSIS_PATH", "")
    parser = argparse.ArgumentParser(description="Run CMSIS-NN float unit-test generation/build flows.")
    parser.add_argument(
        "--tests",
        default="all",
        help=(
            "Comma-separated float test families to run. Supported values include "
            "softmax, activation, reshape, avg_pool, batch_norm, batch_matmul, "
            "concatenation, convolve, depthwise_conv, ds_cnn_s_body, elementwise_add, "
            "elementwise_mul, fully_connected, lstm, max_pool, maximum_minimum, "
            "pad, svdf, transpose, transpose_conv, or all."
        ),
    )
    parser.add_argument(
        "--dtypes",
        default="f32,f16",
        help="Comma-separated dtype list. Supported values: f32, f16.",
    )
    parser.add_argument("--list", action="store_true", help="List supported float test families and exit.")
    parser.add_argument("--generate", action="store_true", help="Generate float reference data.")
    parser.add_argument("--build-host", action="store_true", help="Configure and build host unit tests.")
    parser.add_argument("--run-host", action="store_true", help="Run host unit-test executables.")
    parser.add_argument("--build-cmsis", action="store_true", help="Build CMSIS-Toolbox Corstone-300 test contexts.")
    parser.add_argument("--run-fvp", action="store_true", help="Run CMSIS-Toolbox test images on FVP.")
    parser.add_argument(
        "--cbuild-packs",
        action="store_true",
        help="Pass --packs to cbuild so missing packs are installed automatically.",
    )
    parser.add_argument(
        "--toolchains",
        default="GCC@15.2.1",
        help="Comma-separated CMSIS-Toolbox toolchains, for example GCC@15.2.1 or AC6@6.24.0.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel jobs for cmake/cbuild. FVP test images are still run sequentially.",
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated float samples.")
    parser.add_argument(
        "--cmsis-path",
        default=default_cmsis_path,
        help="CMSIS path passed to the host CMake build. Defaults to the CMSIS_PATH environment variable.",
    )
    parser.add_argument(
        "--host-build-dir",
        default="/tmp/cmsis-nn-float-unit-runner",
        help="Host unit-test build directory.",
    )
    parser.add_argument("--clean-host", action="store_true", help="Delete the host build directory before configuring.")
    parser.add_argument(
        "--fvp-bin",
        help="Path to the Corstone-300 FVP binary. Required when --run-fvp is used.",
    )
    parser.add_argument(
        "--fvp-image-arg",
        default="",
        help="Optional FVP flag used before the image path, for example -a. Defaults to positional image argument.",
    )
    parser.add_argument(
        "--fvp-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each FVP run before the runner is terminated.",
    )
    return parser.parse_args()


def resolve_families(selector: str) -> list[FloatTestFamily]:
    if selector == "all":
        return [FAMILY_CONFIGS[name] for name in sorted(FAMILY_CONFIGS)]

    resolved: list[FloatTestFamily] = []
    seen: set[str] = set()
    for item in parse_csv(selector):
        normalized = FAMILY_ALIASES.get(item, item)
        if normalized not in FAMILY_CONFIGS:
            raise SystemExit(f"Unknown float test family: {item}")
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(FAMILY_CONFIGS[normalized])
    return resolved


def resolve_dtypes(selector: str) -> list[str]:
    dtypes = parse_csv(selector)
    if not dtypes:
        raise SystemExit("At least one dtype must be selected.")
    invalid = [dtype_name for dtype_name in dtypes if dtype_name not in {"f32", "f16"}]
    if invalid:
        raise SystemExit(f"Unsupported dtype selector(s): {', '.join(invalid)}")
    return dtypes


def run_command(args: list[str], cwd: Path) -> None:
    print(f"$ (cd {cwd} && {' '.join(args)})", flush=True)
    subprocess.run(args, cwd=str(cwd), check=True)


def try_command(args: list[str], cwd: Path) -> tuple[bool, str]:
    print(f"$ (cd {cwd} && {' '.join(args)})", flush=True)
    completed = subprocess.run(args, cwd=str(cwd), check=False)
    if completed.returncode == 0:
        return True, ""
    return False, f"exit {completed.returncode}"


def _decode_timeout_output(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def summarize_unity_ticks(output: str) -> str:
    tick_matches = re.findall(r":PASS\s+\((\d+)\s+ticks\)", output)
    if not tick_matches:
        return ""

    tick_values = [int(value) for value in tick_matches]
    total_ticks = sum(tick_values)
    test_count = len(tick_values)
    ticks_label = "tick" if total_ticks == 1 else "ticks"
    tests_label = "test" if test_count == 1 else "tests"
    return f"sum {total_ticks} {ticks_label} ({test_count} {tests_label})"


def try_fvp_command(args: list[str], cwd: Path, timeout_seconds: int) -> tuple[bool, str]:
    print(f"$ (cd {cwd} && {' '.join(args)})", flush=True)
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_timeout_output(exc.stdout)
        stderr_text = _decode_timeout_output(exc.stderr)
        if stdout_text:
            print(stdout_text, end="", flush=True)
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr, flush=True)

        timed_out_after_success = (
            "0 Failures" in stdout_text
            and "\nOK" in stdout_text
            and "FAIL" not in stdout_text
        )
        if timed_out_after_success:
            tick_summary = summarize_unity_ticks(stdout_text)
            detail = f"timeout {timeout_seconds}s after PASS output"
            if tick_summary:
                detail = f"{tick_summary}; {detail}"
            return True, detail
        return False, f"timeout {timeout_seconds}s"

    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)

    output = completed.stdout or ""
    passed = completed.returncode == 0 and "0 Failures" in output and "\nOK" in output and ":FAIL:" not in output
    if passed:
        return True, summarize_unity_ticks(output)

    fail_line = next((line.strip() for line in output.splitlines() if ":FAIL:" in line), "")
    if fail_line:
        return False, fail_line

    if completed.returncode == 0:
        return False, "fvp run failed"
    return False, f"exit {completed.returncode}"


def append_result(
    results: list[StepResult],
    stage: str,
    family: str,
    dtype_name: str = "-",
    toolchain: str = "-",
    status: str = "PASS",
    detail: str = "",
) -> None:
    results.append(
        StepResult(
            stage=stage,
            family=family,
            dtype=dtype_name,
            toolchain=toolchain,
            status=status,
            detail=detail,
        )
    )


def print_summary(results: list[StepResult]) -> None:
    if not results:
        return

    headers = ("Stage", "Family", "DType", "Toolchain", "Status", "Detail")
    rows = [(r.stage, r.family, r.dtype, r.toolchain, r.status, r.detail) for r in results]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(row: tuple[str, str, str, str, str, str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    passed = sum(1 for result in results if result.status == "PASS")
    failed = sum(1 for result in results if result.status == "FAIL")
    skipped = sum(1 for result in results if result.status == "SKIP")

    print("\nFloat Unit-Test Summary")
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)
    print(f"PASS={passed} FAIL={failed} SKIP={skipped}")


def ensure_python_modules(requirements: list[tuple[str, str]], reason: str) -> None:
    missing: list[str] = []
    for module_name, package_name in requirements:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise SystemExit(
            f"Missing Python module(s) required to {reason}: "
            + ", ".join(sorted(missing))
            + ". Install them in the active environment and rerun."
        )


def ensure_host_python_modules() -> None:
    ensure_python_modules(
        [("serial", "pyserial"), ("termcolor", "termcolor")],
        "run the existing host unit-test harness",
    )


def generate_float_test_data(
    families: list[FloatTestFamily], dtypes: list[str], regenerate_input: bool, results: list[StepResult]
) -> bool:
    all_ok = True
    for family in families:
        if family.generator_script is None:
            for dtype_name in dtypes:
                append_result(results, "generate", family.name, dtype_name, status="SKIP", detail="no generator")
            continue
        script_path = UNIT_TEST_ROOT / family.generator_script
        for dtype_name in dtypes:
            family_ok = True
            for selector in family.selector_by_dtype[dtype_name]:
                cmd = [sys.executable, str(script_path), "--dataset", selector]
                if regenerate_input:
                    cmd.append("--regenerate-input")
                if family.name == "fully_connected" and regenerate_input:
                    cmd.extend(["--regenerate-weights", "--regenerate-biases"])
                ok, detail = try_command(cmd, UNIT_TEST_ROOT)
                if not ok:
                    family_ok = False
                    all_ok = False
                    append_result(results, "generate", family.name, dtype_name, status="FAIL", detail=f"{selector} {detail}")
                    break
            if family_ok:
                append_result(results, "generate", family.name, dtype_name)
    return all_ok


def configure_and_build_host(
    families: list[FloatTestFamily],
    dtypes: list[str],
    jobs: int,
    cmsis_path: Path,
    build_dir: Path,
    clean_host: bool,
    results: list[StepResult],
) -> bool:
    ensure_host_python_modules()
    if clean_host and build_dir.exists():
        shutil.rmtree(build_dir)
    if not str(cmsis_path):
        raise SystemExit("--cmsis-path or the CMSIS_PATH environment variable is required for host builds.")

    cmake_args = [
        "cmake",
        "-S",
        str(UNIT_TEST_ROOT),
        "-B",
        str(build_dir),
        "-DBUILD_CMSIS_NN_UNIT=ON",
        f"-DCMSIS_PATH={cmsis_path}",
        f"-DARM_NN_ENABLE_F32={'ON' if 'f32' in dtypes else 'OFF'}",
        f"-DARM_NN_ENABLE_F16={'ON' if 'f16' in dtypes else 'OFF'}",
    ]
    run_command(cmake_args, REPO_ROOT)

    all_ok = True
    for family in families:
        for dtype_name in dtypes:
            target = family.host_target(dtype_name)
            ok, detail = try_command(["cmake", "--build", str(build_dir), "-j", str(jobs), "--target", target], REPO_ROOT)
            if not ok:
                all_ok = False
                append_result(results, "build-host", family.name, dtype_name, status="FAIL", detail=detail)
            else:
                append_result(results, "build-host", family.name, dtype_name)
    return all_ok


def executable_path(build_dir: Path, target_name: str) -> Path:
    target_dir = build_dir / "TestCases" / target_name
    direct_path = target_dir / target_name
    if direct_path.exists():
        return direct_path

    matches = sorted(target_dir.glob(f"{target_name}*"))
    if not matches:
        raise SystemExit(f"Unable to find host executable for target {target_name} in {target_dir}")
    return matches[0]


def run_host_tests(families: list[FloatTestFamily], dtypes: list[str], build_dir: Path, results: list[StepResult]) -> bool:
    all_ok = True
    for family in families:
        for dtype_name in dtypes:
            target_name = family.host_target(dtype_name)
            ok, detail = try_command([str(executable_path(build_dir, target_name))], REPO_ROOT)
            if not ok:
                all_ok = False
                append_result(results, "run-host", family.name, dtype_name, status="FAIL", detail=detail)
            else:
                append_result(results, "run-host", family.name, dtype_name)
    return all_ok


def toolchain_family(toolchain: str) -> str:
    return toolchain.split("@", 1)[0]


def cmsis_image_suffix(toolchain: str) -> str:
    return "axf" if toolchain_family(toolchain) == "AC6" else "elf"


def build_cmsis_tests(
    families: list[FloatTestFamily],
    dtypes: list[str],
    toolchains: list[str],
    jobs: int,
    results: list[StepResult],
    cbuild_packs: bool,
) -> bool:
    all_ok = True
    for toolchain in toolchains:
        for family in families:
            for dtype_name in dtypes:
                context = family.cmsis_context(dtype_name)
                update_cmd = [
                    "cbuild",
                    "--update-rte",
                    "--context",
                    context,
                    str(CMSIS_SOLUTION),
                    "-j",
                    str(jobs),
                    "--toolchain",
                    toolchain,
                ]
                if CMSIS_OUTPUT_ROOT:
                    update_cmd.extend(["--output", CMSIS_OUTPUT_ROOT])
                if cbuild_packs:
                    update_cmd.append("--packs")
                ok, detail = try_command(update_cmd, CMSIS_UNIT_TEST_ROOT)
                if not ok:
                    all_ok = False
                    append_result(results, "build-cmsis", family.name, dtype_name, toolchain, "FAIL", f"update-rte {detail}")
                    continue
                build_cmd = [
                    "cbuild",
                    "--context",
                    context,
                    str(CMSIS_SOLUTION),
                    "-j",
                    str(jobs),
                    "--toolchain",
                    toolchain,
                ]
                if CMSIS_OUTPUT_ROOT:
                    build_cmd.extend(["--output", CMSIS_OUTPUT_ROOT])
                if cbuild_packs:
                    build_cmd.append("--packs")
                ok, detail = try_command(build_cmd, CMSIS_UNIT_TEST_ROOT)
                if not ok:
                    all_ok = False
                    append_result(results, "build-cmsis", family.name, dtype_name, toolchain, "FAIL", detail)
                else:
                    append_result(results, "build-cmsis", family.name, dtype_name, toolchain)
    return all_ok


def cmsis_image_path(family: FloatTestFamily, dtype_name: str, toolchain: str) -> Path:
    family_name = toolchain_family(toolchain)
    binary_name = family.host_target(dtype_name)
    output_root = Path(CMSIS_OUTPUT_ROOT) if CMSIS_OUTPUT_ROOT else CMSIS_UNIT_TEST_ROOT
    return output_root / f"{family.cmsis_context(dtype_name)}-{family_name}" / "outdir" / f"{binary_name}.{cmsis_image_suffix(toolchain)}"


def run_fvp_tests(
    families: list[FloatTestFamily],
    dtypes: list[str],
    toolchains: list[str],
    fvp_bin: Path,
    fvp_image_arg: str,
    fvp_timeout: int,
    results: list[StepResult],
) -> bool:
    if not fvp_bin:
        raise SystemExit("--fvp-bin is required when --run-fvp is used.")

    fvp_common_args = [
        "-C",
        "mps3_board.visualisation.disable-visualisation=1",
        "-C",
        "mps3_board.telnetterminal0.start_telnet=0",
        "-C",
        "mps3_board.uart0.out_file=-",
        "-C",
        "mps3_board.uart0.unbuffered_output=1",
        "-C",
        "mps3_board.uart0.shutdown_on_eot=1",
    ]

    all_ok = True
    for toolchain in toolchains:
        for family in families:
            for dtype_name in dtypes:
                image_path = cmsis_image_path(family, dtype_name, toolchain)
                fvp_cmd = [str(fvp_bin)]
                if fvp_image_arg:
                    fvp_cmd.extend([fvp_image_arg, str(image_path)])
                else:
                    fvp_cmd.append(str(image_path))
                fvp_cmd.extend(fvp_common_args)
                ok, detail = try_fvp_command(fvp_cmd, CMSIS_UNIT_TEST_ROOT, fvp_timeout)
                if not ok:
                    all_ok = False
                    append_result(results, "run-fvp", family.name, dtype_name, toolchain, "FAIL", detail)
                else:
                    append_result(results, "run-fvp", family.name, dtype_name, toolchain, detail=detail)
    return all_ok


def main() -> None:
    args = parse_args()
    if args.list:
        for family_name in sorted(FAMILY_CONFIGS):
            print(family_name)
        return

    if not any((args.generate, args.build_host, args.run_host, args.build_cmsis, args.run_fvp)):
        args.generate = True
        args.build_host = True
        args.run_host = True

    families = resolve_families(args.tests)
    dtypes = resolve_dtypes(args.dtypes)
    toolchains = parse_csv(args.toolchains)
    results: list[StepResult] = []
    failed = False

    if args.generate:
        ensure_python_modules([("numpy", "numpy"), ("torch", "torch")], "generate float reference data")
        failed |= not generate_float_test_data(families, dtypes, args.regenerate_input, results)
    if args.build_host:
        failed |= not configure_and_build_host(
            families,
            dtypes,
            args.jobs,
            Path(args.cmsis_path),
            Path(args.host_build_dir),
            args.clean_host,
            results,
        )
    if args.run_host:
        failed |= not run_host_tests(families, dtypes, Path(args.host_build_dir), results)
    if args.build_cmsis:
        failed |= not build_cmsis_tests(families, dtypes, toolchains, args.jobs, results, args.cbuild_packs)
    if args.run_fvp:
        failed |= not run_fvp_tests(
            families,
            dtypes,
            toolchains,
            Path(args.fvp_bin) if args.fvp_bin else None,
            args.fvp_image_arg,
            args.fvp_timeout,
            results,
        )

    print_summary(results)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
