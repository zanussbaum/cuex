# cuex

**CUDA Execute** — Run CUDA/C++ code on remote GPUs via Modal.

## Overview

A CLI tool for compiling and executing CUDA/C++ code on remote GPU infrastructure. Write kernels locally, run them in the cloud.

**Develop locally, test remotely.** Write and iterate on CUDA/C++ code on your local machine, then seamlessly run it on real GPU hardware in the cloud. Modal provides the sandboxed execution environment, so you can safely test code without worrying about breaking anything locally.

## Installation

### As a uv tool (recommended)

```bash
# Install directly from GitHub
uv tool install git+https://github.com/zanussbaum/cuex

# Verify installation
cuex --help
```

### Modal Setup

cuex requires a [Modal](https://modal.com) account (free tier available).

```bash
# Authenticate with Modal (one-time)
modal setup
```

## Usage

### Run a program

```bash
cuex run <source_file> [options] [-- program_args...]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu`, `-g` | GPU type: none, T4, L4, A10G, A100, H100 | Auto-detect from extension |
| `--timeout`, `-t` | Max execution time in seconds | 300 |
| `--asm` | Generate ASM/PTX/SASS output | false |
| `--data`, `-d` | Glob pattern for data files to upload (repeatable) | - |

**Auto-detection:**
- `.cpp` files → CPU-only execution
- `.cu` files → GPU execution (default: L4)

### Examples

```bash
# Run CUDA code on default GPU (L4)
cuex run my_kernel.cu

# Run CUDA on A100
cuex run my_kernel.cu --gpu A100

# Run C++ on CPU only
cuex run my_program.cpp

# Get PTX/SASS output
cuex run my_kernel.cu --asm

# Upload data files with your kernel
cuex run matmul.cu --data "*.bin" --data "config.json"

# Pass arguments to the compiled program
cuex run mandelbrot.cpp -- --width 1024 --height 768

# Check version
cuex --version
```

### Compare runs

```bash
# Show code diff between last two runs
cuex diff my_kernel.cu

# Compare specific runs
cuex diff my_kernel.cu -n 2  # Compare runs 2 and 3
```

## Output

### Streaming

Compile and execute logs stream to stdout in real-time:

```
$ cuex run my_kernel.cu
Source: my_kernel.cu
GPU: L4
Timeout: 300s

[compile] nvcc -O3 -o /tmp/program /tmp/my_kernel.cu
[execute] /tmp/program
[execute] Kernel executed in 0.042ms
[execute] Result: 42
✓ Job complete (exit code 0)
Output saved to: ./cuex-out/my_kernel/L4/20250111_143052_a1b2c3d4/
```

### File Output

Logs are saved to disk organized by file and GPU type:

```
./cuex-out/
└── <filename>/
    └── <gpu_type>/
        └── <job_id>/
            ├── compile_log.txt
            ├── execute_log.txt
            ├── source.cu
            ├── asm_output.txt     # If --asm used
            └── out/               # Program output files
```

## Requirements

- Python 3.11+
- Modal account (free tier available)

## License

MIT
