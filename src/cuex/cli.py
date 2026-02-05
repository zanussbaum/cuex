"""CLI entry point for cuex."""

import glob
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from . import __version__
from .output import OutputManager
from .runner import GPUType, compile_and_run


def collect_data_files(patterns: list[str], source_dir: Path) -> tuple[dict[str, bytes], int]:
    """Collect data files matching glob patterns.

    Args:
        patterns: List of glob patterns (e.g., ["*.bin", "data/*.json"])
        source_dir: Directory of the source file (for computing relative paths)

    Returns:
        Tuple of (dict of relative path -> bytes, total bytes)
    """
    data_files: dict[str, bytes] = {}
    total_bytes = 0
    cwd = Path.cwd()

    for pattern in patterns:
        # Resolve pattern from CWD (so explicit paths like "kernels/lab4/*.bin" work)
        full_pattern = str(cwd / pattern)
        matches = glob.glob(full_pattern, recursive=True)

        for match in matches:
            match_path = Path(match).resolve()
            if match_path.is_file():
                # Store relative to source directory (so files end up as siblings)
                try:
                    rel_path = match_path.relative_to(source_dir)
                except ValueError:
                    # File is outside source dir, just use the filename
                    rel_path = Path(match_path.name)

                content = match_path.read_bytes()
                data_files[str(rel_path)] = content
                total_bytes += len(content)

    return data_files, total_bytes


def collect_local_includes(
    source_path: Path,
    base_dir: Path | None = None,
    seen: set[Path] | None = None,
) -> dict[str, bytes]:
    """Recursively collect local #include "..." dependencies.

    Args:
        source_path: Path to the source file to scan
        base_dir: Base directory for computing relative paths (defaults to source_path.parent)
        seen: Set of already-processed files (to avoid cycles)

    Returns:
        Dict of relative path -> file content (as bytes)

    Raises:
        typer.Exit: If an included file cannot be found
    """
    if seen is None:
        seen = set()
    if base_dir is None:
        base_dir = source_path.parent.resolve()

    source_path = source_path.resolve()
    if source_path in seen:
        return {}
    seen.add(source_path)

    includes: dict[str, bytes] = {}
    content = source_path.read_text()

    # Match #include "filename" (not <filename>)
    # This regex handles optional whitespace and captures the path
    include_pattern = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)

    for match in include_pattern.finditer(content):
        include_name = match.group(1)

        # Resolve relative to the file containing the include
        include_path = (source_path.parent / include_name).resolve()

        if not include_path.exists():
            console.print(f"[red]Error:[/red] Cannot find included file: {include_name}")
            console.print(f"  Referenced from: {source_path}")
            console.print(f"  Looked for: {include_path}")
            raise typer.Exit(1)

        if include_path in seen:
            continue

        # Compute path relative to base_dir
        try:
            rel_path = include_path.relative_to(base_dir)
        except ValueError:
            # File is outside base_dir (e.g., ../common/header.h)
            console.print(
                f"[red]Error:[/red] Included file is outside source directory: {include_name}"
            )
            console.print(f"  Source dir: {base_dir}")
            console.print(f"  Include path: {include_path}")
            console.print("  Hint: Move shared headers into the source directory")
            raise typer.Exit(1)

        includes[str(rel_path)] = include_path.read_bytes()

        # Recursively collect includes from this file
        nested = collect_local_includes(include_path, base_dir, seen)
        includes.update(nested)

    return includes


def print_colored_diff(console: Console, diff_text: str) -> None:
    """Print a unified diff with colors."""
    for line in diff_text.splitlines():
        escaped_line = escape(line)
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[bold]{escaped_line}[/bold]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{escaped_line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{escaped_line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{escaped_line}[/red]")
        else:
            console.print(escaped_line)


def version_callback(value: bool) -> None:
    if value:
        print(f"cuex {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="cuex",
    help="Run CUDA/C++ code on remote GPUs via Modal.",
    no_args_is_help=True,
)
console = Console()


def detect_gpu_from_extension(filename: str) -> GPUType:
    """Auto-detect GPU type based on file extension."""
    if filename.endswith(".cu"):
        return GPUType.L4
    return GPUType.NONE


@app.callback(invoke_without_command=True)
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Run CUDA/C++ code on remote GPUs via Modal."""
    pass


@app.command(
    context_settings={"allow_interspersed_args": True},
)
def run(
    source_file: Path = typer.Argument(
        ...,
        help="Source file to compile and run (.cpp or .cu)",
        exists=True,
        readable=True,
    ),
    program_args: list[str] = typer.Argument(
        None,
        help="Arguments to pass to the compiled program (after --)",
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        "-g",
        help="GPU type: none, T4, L4, A10G, A100, H100, B200 (auto-detected from extension if not specified)",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Max execution time in seconds",
    ),
    asm: bool = typer.Option(
        False,
        "--asm",
        help="Generate ASM/PTX/SASS output",
    ),
    data: list[str] = typer.Option(
        None,
        "--data",
        "-d",
        help="Glob pattern for data files to upload (can be specified multiple times)",
    ),
    cflags: list[str] = typer.Option(
        None,
        "--cflags",
        "-c",
        help="Extra compiler flags (can be specified multiple times, e.g. --cflags=-DDEBUG --cflags=-Wno-deprecated)",
    ),
) -> None:
    """Compile and run a CUDA/C++ source file on remote GPU infrastructure."""
    # Validate file extension
    if source_file.suffix not in (".cpp", ".cu"):
        console.print(f"[red]Error:[/red] Unsupported file type: {source_file.suffix}")
        console.print("Supported extensions: .cpp, .cu")
        raise typer.Exit(1)

    # Determine GPU type
    if gpu is None:
        gpu_type = detect_gpu_from_extension(source_file.name)
    else:
        try:
            gpu_type = GPUType[gpu.upper()]
        except KeyError:
            console.print(f"[red]Error:[/red] Invalid GPU type: {gpu}")
            console.print(f"Valid options: {', '.join(g.name for g in GPUType)}")
            raise typer.Exit(1) from None

    # Read source code
    source_code = source_file.read_text()
    base_dir = source_file.parent.resolve()

    # Collect local includes (headers referenced via #include "...")
    include_files = collect_local_includes(source_file, base_dir)

    # Collect data files
    data_files: dict[str, bytes] = {}
    data_bytes = 0
    if data:
        data_files, data_bytes = collect_data_files(data, base_dir)
        if not data_files:
            console.print(f"[yellow]Warning:[/yellow] No files matched patterns: {data}")

    # Merge includes and data files (includes take precedence if overlap)
    all_files = {**data_files, **include_files}

    console.print(f"[blue]Source:[/blue] {source_file}")
    console.print(f"[blue]GPU:[/blue] {gpu_type.name}")
    console.print(f"[blue]Timeout:[/blue] {timeout}s")
    if asm:
        console.print("[blue]ASM output:[/blue] enabled")
    if include_files:
        console.print(f"[blue]Includes:[/blue] {len(include_files)} files")
        for rel_path in sorted(include_files.keys()):
            console.print(f"  [dim]{rel_path}[/dim]")
    if data_files:
        console.print(
            f"[blue]Data files:[/blue] {len(data_files)} files ({data_bytes / 1024:.1f} KB)"
        )
        for rel_path in sorted(data_files.keys()):
            console.print(f"  [dim]{rel_path}[/dim]")
    if cflags:
        console.print(f"[blue]Extra cflags:[/blue] {' '.join(cflags)}")
    console.print()

    # Filter out the "--" separator if present
    args = [arg for arg in (program_args or []) if arg != "--"]

    # Reconstruct the command for logging
    cmd_parts = ["cuex", "run"]
    if gpu is not None:
        cmd_parts.extend(["--gpu", gpu])
    if timeout != 300:
        cmd_parts.extend(["--timeout", str(timeout)])
    if asm:
        cmd_parts.append("--asm")
    if data:
        for pattern in data:
            cmd_parts.extend(["--data", pattern])
    if cflags:
        for flag in cflags:
            cmd_parts.extend(["--cflags", flag])
    cmd_parts.append(str(source_file))
    if args:
        cmd_parts.append("--")
        cmd_parts.extend(args)
    command = " ".join(cmd_parts)

    # Run the compilation and execution
    try:
        result = compile_and_run(
            source_code=source_code,
            filename=source_file.name,
            gpu_type=gpu_type,
            timeout=timeout,
            generate_asm=asm,
            program_args=args,
            data_files=all_files,
            extra_compile_flags=cflags,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Save output files
    output_manager = OutputManager()
    output_dir = output_manager.save_job_output(
        job_id=result["job_id"],
        source_code=source_code,
        source_filename=source_file.name,
        gpu_type=gpu_type.name,
        compile_log=result["compile_log"],
        execute_log=result["execute_log"],
        asm_output=result.get("asm_output"),
        output_files=result.get("output_files", {}),
        command=command,
    )

    # Show diff from previous run if code changed
    diff_text, newer_run, older_run = output_manager.diff_runs(
        source_filename=source_file.name,
        gpu_type=gpu_type.name,
        run_index=1,
    )
    if diff_text and diff_text != "":
        console.print()
        console.print("[blue]Changes from previous run:[/blue]")
        console.print(f"[dim]{older_run.name} → {newer_run.name}[/dim]")
        print_colored_diff(console, diff_text)
        console.print()

    # Print summary
    if result["success"]:
        console.print(f"[green]\u2713[/green] Job complete (exit code {result['exit_code']})")
    else:
        console.print(f"[red]\u2717[/red] Job failed (exit code {result['exit_code']})")

    console.print(f"[dim]Output saved to: {output_dir}[/dim]")

    raise typer.Exit(result["exit_code"])


@app.command()
def diff(
    source_file: Path = typer.Argument(
        ...,
        help="Source file to diff runs for (.cpp or .cu)",
        exists=True,
        readable=True,
    ),
    n: int = typer.Option(
        1,
        "--n",
        "-n",
        help="Compare run N with run N+1 (1 = compare last two runs)",
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        "-g",
        help="GPU type: none, T4, L4, A10G, A100, H100, B200 (auto-detected from extension if not specified)",
    ),
) -> None:
    """Show code diff between runs of a source file."""
    # Validate file extension
    if source_file.suffix not in (".cpp", ".cu"):
        console.print(f"[red]Error:[/red] Unsupported file type: {source_file.suffix}")
        raise typer.Exit(1)

    # Determine GPU type
    if gpu is None:
        gpu_type = detect_gpu_from_extension(source_file.name)
    else:
        try:
            gpu_type = GPUType[gpu.upper()]
        except KeyError:
            console.print(f"[red]Error:[/red] Invalid GPU type: {gpu}")
            raise typer.Exit(1) from None

    output_manager = OutputManager()
    diff_text, newer_run, older_run = output_manager.diff_runs(
        source_filename=source_file.name,
        gpu_type=gpu_type.name,
        run_index=n,
    )

    if diff_text is None:
        runs = output_manager.get_runs_for_file(source_file.name, gpu_type.name)
        if len(runs) == 0:
            console.print(
                f"[yellow]No runs found for {source_file.name} on {gpu_type.name}[/yellow]"
            )
        elif len(runs) < n + 1:
            console.print(
                f"[yellow]Not enough runs to diff (found {len(runs)}, need {n + 1})[/yellow]"
            )
        raise typer.Exit(1)

    if diff_text == "":
        console.print("[dim]No code changes between runs (whitespace only)[/dim]")
        console.print(f"  [dim]{older_run.name} → {newer_run.name}[/dim]")
        raise typer.Exit(0)

    console.print(f"[blue]Diff:[/blue] {source_file.name} on {gpu_type.name}")
    console.print(f"[dim]{older_run.name} → {newer_run.name}[/dim]")
    console.print()
    print_colored_diff(console, diff_text)


if __name__ == "__main__":
    app()
