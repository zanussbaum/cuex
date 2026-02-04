"""Modal function definitions for compiling and running code."""

import subprocess
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import modal
from rich.console import Console

console = Console()

# Define the Modal app
app = modal.App("cuex")

# Define the CUDA image with CUTLASS for tensor core development
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("g++", "git")
    .run_commands(
        "git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass"
    )
    .uv_pip_install("rich")
)

cpu_image = modal.Image.debian_slim(python_version="3.11").apt_install("g++").uv_pip_install("rich")


class GPUType(str, Enum):
    """Available GPU types."""

    NONE = "none"
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    A100 = "A100"
    H100 = "H100"
    B200 = "B200"


# Mapping from GPU type to CUDA compute capability
GPU_ARCH_FLAGS: dict[GPUType, list[str]] = {
    GPUType.NONE: [],
    GPUType.T4: ["-gencode", "arch=compute_75,code=sm_75"],  # Turing
    GPUType.L4: ["-gencode", "arch=compute_89,code=sm_89"],  # Ada Lovelace
    GPUType.A10G: ["-gencode", "arch=compute_86,code=sm_86"],  # Ampere
    GPUType.A100: ["-gencode", "arch=compute_80,code=sm_80"],  # Ampere
    GPUType.H100: ["-gencode", "arch=compute_90a,code=sm_90a"],  # Hopper (90a for wgmma/TMA)
    GPUType.B200: ["-gencode", "arch=compute_100a,code=sm_100a"],  # Blackwell (100a for tcgen05)
}


def get_modal_gpu(gpu_type: GPUType) -> str | None:
    """Convert GPUType to Modal GPU specification."""
    if gpu_type == GPUType.NONE:
        return None
    return gpu_type.value


def generate_job_id() -> str:
    """Generate a unique job ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hex = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_hex}"


@app.function(image=cuda_image, gpu="T4", timeout=300)
def _run_with_t4(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on T4 GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.T4]
    )


@app.function(image=cuda_image, gpu="L4", timeout=300)
def _run_with_l4(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on L4 GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.L4]
    )


@app.function(image=cuda_image, gpu="A10G", timeout=300)
def _run_with_a10g(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on A10G GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.A10G]
    )


@app.function(image=cuda_image, gpu="A100", timeout=300)
def _run_with_a100(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on A100 GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.A100]
    )


@app.function(image=cuda_image, gpu="H100!", timeout=300)
def _run_with_h100(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on H100 GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.H100]
    )


@app.function(image=cuda_image, gpu="B200", timeout=300)
def _run_with_b200(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code on B200 GPU."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.B200]
    )


@app.function(image=cpu_image, timeout=300)
def _run_with_cpu(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
) -> dict[str, Any]:
    """Run code without GPU (CPU only)."""
    return _compile_and_run_impl(
        source_code, filename, generate_asm, program_args, data_files, GPU_ARCH_FLAGS[GPUType.NONE]
    )


def detect_cuda_libraries(source_code: str, data_files: dict[str, bytes] | None = None) -> list[str]:
    """Auto-detect required CUDA libraries based on #include directives.

    Scans the main source code and any included header files (data_files).
    """
    library_map = {
        "cuda.h": "-lcuda",  # CUDA driver API (TMA, etc.)
        "cublas_v2.h": "-lcublas",
        "cublas.h": "-lcublas",
        "cublasLt.h": "-lcublasLt",
        "cufft.h": "-lcufft",
        "cusparse.h": "-lcusparse",
        "curand.h": "-lcurand",
        "cusolver_common.h": "-lcusolver",
        "cusolverDn.h": "-lcusolver",
    }

    # Collect all source text to scan (main file + headers)
    all_sources = [source_code]
    if data_files:
        for content in data_files.values():
            try:
                all_sources.append(content.decode("utf-8"))
            except UnicodeDecodeError:
                pass  # Skip binary files

    libs = []
    for source in all_sources:
        for header, lib_flag in library_map.items():
            has_include = f"#include <{header}>" in source or f'#include "{header}"' in source
            if has_include and lib_flag not in libs:
                libs.append(lib_flag)
    return libs


def _compile_and_run_impl(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str],
    data_files: dict[str, bytes],
    arch_flags: list[str] | None = None,
) -> dict[str, Any]:
    """Implementation of compile and run logic (runs inside Modal container)."""
    from rich.console import Console

    console = Console()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        source_path = tmppath / filename
        binary_path = tmppath / "program"
        out_dir = tmppath / "out"
        out_dir.mkdir()

        # Write source file
        source_path.write_text(source_code)

        # Write data files
        for rel_path, content in data_files.items():
            data_path = tmppath / rel_path
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(content)
            console.print(f"[data] wrote {rel_path} ({len(content)} bytes)")

        # Determine compiler and flags based on extension
        is_cuda = filename.endswith(".cu")

        if is_cuda:
            # Auto-detect required CUDA libraries (scan main file + headers)
            cuda_libs = detect_cuda_libraries(source_code, data_files)

            # Check if CUTLASS headers are used
            uses_cutlass = any(
                inc in source_code
                for inc in ["cute/", "cutlass/", "<cute", "<cutlass"]
            )

            compile_cmd = [
                "nvcc",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--std=c++20",
                "-Xptxas",
                "-v",
            ]

            # Add CUTLASS include paths if needed
            if uses_cutlass:
                compile_cmd.extend([
                    "-I/opt/cutlass/include",
                    "-I/opt/cutlass/tools/util/include",
                ])

            if arch_flags:
                compile_cmd.extend(arch_flags)
            compile_cmd.extend(["-o", str(binary_path), str(source_path)])
            compile_cmd.extend(cuda_libs)

            asm_cmd = [
                "nvcc",
                "-ptx",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--std=c++20",
            ]
            if uses_cutlass:
                asm_cmd.extend([
                    "-I/opt/cutlass/include",
                    "-I/opt/cutlass/tools/util/include",
                ])
            if arch_flags:
                asm_cmd.extend(arch_flags)
            asm_cmd.extend(["-o", str(tmppath / "program.ptx"), str(source_path)])
        else:
            compile_cmd = [
                "g++",
                "-O3",
                "-march=native",
                "-Wall",
                "-Wextra",
                "-o",
                str(binary_path),
                str(source_path),
            ]
            asm_cmd = [
                "g++",
                "-S",
                "-O3",
                "-march=native",
                "-o",
                str(tmppath / "program.s"),
                str(source_path),
            ]

        # Compile
        console.print(f"[compile] {' '.join(compile_cmd)}")
        compile_result = subprocess.run(
            compile_cmd, capture_output=True, text=True, cwd=str(tmppath)
        )
        compile_log = compile_result.stdout + compile_result.stderr

        if compile_log.strip():
            for line in compile_log.strip().split("\n"):
                console.print(f"[compile] {line}")

        if compile_result.returncode != 0:
            return {
                "success": False,
                "compile_log": compile_log,
                "execute_log": "",
                "asm_output": None,
                "output_files": {},
                "exit_code": compile_result.returncode,
            }

        # Generate ASM if requested
        asm_output = None
        if generate_asm:
            console.print(f"[asm] {' '.join(asm_cmd)}")
            asm_result = subprocess.run(asm_cmd, capture_output=True, text=True, cwd=str(tmppath))
            if asm_result.returncode == 0:
                asm_file = tmppath / ("program.ptx" if is_cuda else "program.s")
                if asm_file.exists():
                    asm_output = asm_file.read_text()

            # For CUDA, also generate SASS
            if is_cuda and binary_path.exists():
                sass_cmd = [
                    "cuobjdump",
                    "-sass",
                    str(binary_path),
                ]
                sass_result = subprocess.run(
                    sass_cmd, capture_output=True, text=True, cwd=str(tmppath)
                )
                if sass_result.returncode == 0 and sass_result.stdout:
                    asm_output = f"=== PTX ===\n{asm_output}\n\n=== SASS ===\n{sass_result.stdout}"

        # Execute
        # Run from tmppath so programs that write to "out/..." work correctly
        exec_cmd = [str(binary_path), *program_args]
        console.print(f"[execute] {' '.join(exec_cmd)}")
        execute_result = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            cwd=str(tmppath),
        )
        execute_log = execute_result.stdout + execute_result.stderr

        if execute_log.strip():
            for line in execute_log.strip().split("\n"):
                console.print(f"[execute] {line}")

        # Collect output files
        output_files: dict[str, bytes] = {}
        for output_file in out_dir.rglob("*"):
            if output_file.is_file():
                rel_path = output_file.relative_to(out_dir)
                output_files[str(rel_path)] = output_file.read_bytes()

        return {
            "success": execute_result.returncode == 0,
            "compile_log": compile_log,
            "execute_log": execute_log,
            "asm_output": asm_output,
            "output_files": output_files,
            "exit_code": execute_result.returncode,
        }


def compile_and_run(
    source_code: str,
    filename: str,
    gpu_type: GPUType,
    timeout: int = 300,
    generate_asm: bool = False,
    program_args: list[str] | None = None,
    data_files: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Compile and run code on remote GPU infrastructure.

    Args:
        source_code: The source code to compile and run.
        filename: The filename (used to determine compiler).
        gpu_type: The GPU type to use.
        timeout: Max execution time in seconds.
        generate_asm: Whether to generate assembly output.
        program_args: Arguments to pass to the compiled program.
        data_files: Dict of relative path -> bytes for data files to upload.

    Returns:
        A dictionary containing:
        - success: Whether the job completed successfully
        - job_id: Unique identifier for this job
        - compile_log: Compiler stdout/stderr
        - execute_log: Program stdout/stderr
        - asm_output: Assembly output (if requested)
        - output_files: Dict of filename -> bytes for files in ./out/
        - exit_code: The exit code of the program
    """
    job_id = generate_job_id()

    # Select the appropriate function based on GPU type
    gpu_functions = {
        GPUType.NONE: _run_with_cpu,
        GPUType.T4: _run_with_t4,
        GPUType.L4: _run_with_l4,
        GPUType.A10G: _run_with_a10g,
        GPUType.A100: _run_with_a100,
        GPUType.H100: _run_with_h100,
        GPUType.B200: _run_with_b200,
    }

    run_func = gpu_functions[gpu_type]

    # Call the Modal function
    with modal.enable_output(), app.run():
        result = run_func.remote(
            source_code, filename, generate_asm, program_args or [], data_files or {}
        )

    result["job_id"] = job_id
    return result
