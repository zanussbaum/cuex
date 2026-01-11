# cuex

**CUDA Execute** — Run CUDA/C++ code on remote GPUs via Modal.

## Overview

A CLI tool for compiling and executing CUDA/C++ code on remote GPU infrastructure. Write kernels locally, run them in the cloud.

## Vision

**Develop locally, test remotely.** Write and iterate on CUDA/C++ code on your local machine, then seamlessly run it on real GPU hardware in the cloud. Modal provides the sandboxed execution environment, so you can safely test code without worrying about breaking anything locally.

Ideal workflow:
1. Edit code in your favorite editor
2. Run `cuex kernel.cu` 
3. See streaming output in your terminal
4. Iterate

## Goals

1. **Simple CLI**: `cuex my_program.cu`
2. **Support for C++ and CUDA** compilation and execution
3. **Streaming output** — see compile/execute logs in real-time
4. **Local output files** for logs and program artifacts
5. **GPU selection** with sensible defaults
6. **Minimal setup** — installable as a `uv` tool

## Non-Goals

- Authentication/multi-user system (Modal handles this)
- Job queuing beyond what Modal provides
- Sandboxing (Modal handles container isolation)

## CLI Interface

### Run a program

```bash
cuex run <source_file> [options] [-- program_args...]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--gpu`, `-g` | GPU type (none, T4, A10G, A100, H100) | Auto-detect from file extension |
| `--timeout`, `-t` | Max execution time in seconds | 300 |
| `--asm` | Generate ASM/PTX/SASS output | false |
| `--version`, `-V` | Show version and exit | - |

**Auto-detection rules:**
- `.cpp` files → CPU-only execution
- `.cu` files → GPU execution (default: T4)

### Examples

```bash
# Run CUDA code on default GPU (T4)
cuex run my_kernel.cu

# Run CUDA on A100
cuex run my_kernel.cu --gpu A100

# Run C++ on CPU only
cuex run my_program.cpp

# Get PTX/SASS output
cuex run my_kernel.cu --asm

# Pass arguments to the compiled program
cuex run mandelbrot.cpp -- -i vector -r 512

# Check version
cuex --version
```

## Output Behavior

### Streaming

Compile and execute logs stream to stdout in real-time as the job runs. You see output as it happens, not after the job completes.

```
$ cuex run my_kernel.cu
[compile] nvcc -O3 -o /tmp/program /tmp/my_kernel.cu
[execute] /tmp/program
[execute] Kernel executed in 0.042ms
[execute] Result: 42
✓ Job complete (exit code 0)
```

### File Output

Logs are saved to disk organized by file and GPU type for easy comparison:

```
./cuex-out/
└── <filename>/              # Source file (without extension)
    └── <gpu_type>/          # GPU used (T4, A100, NONE, etc.)
        └── <job_id>/        # Timestamp + random hex
            ├── compile_log.txt
            ├── execute_log.txt
            ├── source.cpp|cu
            ├── asm_output.txt     # If --asm used
            └── out/               # Program output files
```

**Example:**
```
./cuex-out/
└── mandelbrot_gpu/
    ├── T4/
    │   └── 20241215_143052_a1b2c3d4/
    └── A100/
        └── 20241215_144012_b2c3d4e5/
```

**Job ID format:** `<timestamp>_<random_hex>` (e.g., `20241215_143052_a1b2c3d4`)

## Technical Design

### Modal Image

Base image: `nvidia/cuda:12.2.0-devel-ubuntu22.04`

Installed packages:
- `g++` (for C++ compilation)
- `nvcc` (included in CUDA image)

### Modal Function

```python
@app.function(
    gpu=<selected_gpu>,
    image=cuda_image,
    timeout=<timeout>
)
def compile_and_run(
    source_code: str,
    filename: str,
    generate_asm: bool,
    program_args: list[str]
) -> dict:
    # 1. Write source to /tmp/<filename>
    # 2. Compile (g++ or nvcc depending on extension)
    # 3. If --asm, generate assembly output
    # 4. Execute compiled binary with program_args
    # 5. Collect any files from ./out/
    # 6. Return logs and output files
```

### Return Value

```python
{
    "success": bool,
    "job_id": str,
    "compile_log": str,
    "execute_log": str,
    "asm_output": str | None,      # If --asm flag used
    "output_files": dict[str, bytes],  # filename -> contents
    "exit_code": int,
}
```

## Compilation Commands

### C++ (`.cpp`)
```bash
g++ -O3 -march=native -o /tmp/program /tmp/<filename>
```

### CUDA (`.cu`)
```bash
nvcc -O3 -o /tmp/program /tmp/<filename>
```

### ASM Output

**C++ (x86-64 assembly):**
```bash
g++ -S -O3 -march=native -o /tmp/program.s /tmp/<filename>
```

**CUDA (PTX):**
```bash
nvcc -ptx -o /tmp/program.ptx /tmp/<filename>
```

**CUDA (SASS):**
```bash
cuobjdump -sass /tmp/program > /tmp/program.sass
```

### Custom Compiler Arguments

> **Note:** We may want to support custom compiler flags in the future (e.g., `-arch=sm_80` for specific CUDA architectures, `-g` for debug symbols, custom include paths).
>
> This doesn't change the architecture — just add an optional `--cflags` argument that appends to the compile command. Not implementing in V1, but the design accommodates it.

## Installation & Setup

### As a uv tool (recommended)

```bash
# Install directly from GitHub
uv tool install git+https://github.com/zanussbaum/cuex

# Or clone and install locally
git clone https://github.com/zanussbaum/cuex
cd cuex
uv tool install .

# Verify installation
cuex --help
```

### Modal Setup

```bash
# Install modal (if not using uv tool)
uv add modal
# or: pip install modal

# Authenticate with Modal (one-time)
modal setup
```

### Project Structure

```
cuex/
├── pyproject.toml        # Package metadata, dependencies, entry points
├── README.md
├── src/
│   └── cuex/
│       ├── __init__.py
│       ├── cli.py        # CLI entry point (typer)
│       ├── runner.py     # Modal function definitions
│       └── output.py     # Local file output handling
└── tests/
    └── ...
```

### pyproject.toml (key sections)

```toml
[project]
name = "cuex"
version = "0.1.0"
description = "Run CUDA/C++ code on remote GPUs via Modal"
requires-python = ">=3.11"
dependencies = [
    "modal>=0.64.0",
    "typer>=0.9.0",
    "rich>=13.0.0",  # For nice terminal output
]

[project.scripts]
cuex = "cuex.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Dependencies

**User's machine:**
- Python 3.11+
- `modal` package
- Modal account (free tier available)

**Remote (Modal container):**
- CUDA 12.2 toolkit
- g++ compiler

## Future Enhancements (Out of Scope for V1)

- [ ] Kernel resource usage in compile log (`-Xptxas=-v` for registers, spills, shared memory)
- [ ] Include program args in output folder structure (for comparing different arg combinations)
- [ ] Multiple source files / includes
- [ ] Custom compiler flags (`--cflags`)
- [ ] Interactive mode (REPL-style)
- [ ] Persistent storage for large datasets (Modal Volumes)
- [ ] Benchmark comparison mode
- [ ] Support for other languages (Rust, etc.)
- [ ] Watch mode (`cuex --watch kernel.cu` — rerun on file change)
- [ ] Config file for project defaults (`.cuex.toml`)

## Open Questions

1. ~~**Streaming output:** Show compile/execute logs in real-time vs. wait for completion?~~ → **Streaming**
2. **Output storage:** Keep all runs vs. overwrite? → Start with keep all, revisit later
3. **GPU default for CUDA:** T4 (cheapest) or something more powerful? → T4 for now
4. **Memory limits:** Should we expose Modal's `memory` parameter? → Not in V1

---

## Status

- [x] Spec drafted
- [x] uv tool structure defined
- [x] Implementation started
- [x] Basic CLI working
- [ ] Testing complete
- [ ] Published to GitHub
