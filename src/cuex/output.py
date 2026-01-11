"""Local file output handling for cuex."""

import difflib
import json
from pathlib import Path


class OutputManager:
    """Manages local output file storage for job results."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize the output manager.

        Args:
            base_dir: Base directory for output files. Defaults to ./cuex-out/
        """
        self.base_dir = base_dir or Path.cwd() / "cuex-out"

    def save_job_output(
        self,
        job_id: str,
        source_code: str,
        source_filename: str,
        gpu_type: str,
        compile_log: str,
        execute_log: str,
        asm_output: str | None = None,
        output_files: dict[str, bytes] | None = None,
        command: str | None = None,
    ) -> Path:
        """Save all output from a job to disk.

        Args:
            job_id: Unique identifier for this job.
            source_code: The submitted source code.
            source_filename: Original filename of the source.
            gpu_type: GPU type used for execution (e.g., "T4", "NONE").
            compile_log: Compiler stdout/stderr.
            execute_log: Program stdout/stderr.
            asm_output: Assembly output (if generated).
            output_files: Dict of filename -> bytes for program output files.
            command: The full command used to invoke this run.

        Returns:
            Path to the job output directory.
        """
        # Structure: <base>/<filename_without_ext>/<gpu_type>/<job_id>/
        file_stem = Path(source_filename).stem
        job_dir = self.base_dir / file_stem / gpu_type / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save source file
        source_path = job_dir / source_filename
        source_path.write_text(source_code)

        # Save compile log
        compile_log_path = job_dir / "compile_log.txt"
        compile_log_path.write_text(compile_log)

        # Save execute log
        execute_log_path = job_dir / "execute_log.txt"
        execute_log_path.write_text(execute_log)

        # Save ASM output if present
        if asm_output:
            asm_path = job_dir / "asm_output.txt"
            asm_path.write_text(asm_output)

        # Save program output files
        if output_files:
            out_dir = job_dir / "out"
            out_dir.mkdir(exist_ok=True)
            for filename, contents in output_files.items():
                file_path = out_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(contents)

        # Save metadata
        meta = {
            "job_id": job_id,
            "source_filename": source_filename,
            "gpu_type": gpu_type,
        }
        if command:
            meta["command"] = command
        meta_path = job_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return job_dir

    def _get_all_job_dirs(self) -> list[Path]:
        """Get all job directories across all files and GPU types.

        Returns:
            List of job directory paths, sorted by name (timestamp) descending.
        """
        if not self.base_dir.exists():
            return []

        job_dirs = []
        # Structure: base_dir/<file_stem>/<gpu_type>/<job_id>/
        for file_dir in self.base_dir.iterdir():
            if not file_dir.is_dir():
                continue
            for gpu_dir in file_dir.iterdir():
                if not gpu_dir.is_dir():
                    continue
                for job_dir in gpu_dir.iterdir():
                    if job_dir.is_dir():
                        job_dirs.append(job_dir)

        return sorted(job_dirs, key=lambda p: p.name, reverse=True)

    def get_latest_job(self) -> Path | None:
        """Get the path to the most recent job output directory.

        Returns:
            Path to the latest job directory, or None if no jobs exist.
        """
        job_dirs = self._get_all_job_dirs()
        return job_dirs[0] if job_dirs else None

    def list_jobs(self, limit: int = 10) -> list[Path]:
        """List recent job output directories.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            List of paths to job directories, most recent first.
        """
        return self._get_all_job_dirs()[:limit]

    def cleanup_old_jobs(self, keep: int = 10) -> int:
        """Remove old job directories, keeping the most recent ones.

        Args:
            keep: Number of recent jobs to keep.

        Returns:
            Number of jobs removed.
        """
        import shutil

        job_dirs = self._get_all_job_dirs()
        to_remove = job_dirs[keep:]

        for job_dir in to_remove:
            shutil.rmtree(job_dir)
            # Clean up empty parent directories
            gpu_dir = job_dir.parent
            if gpu_dir.exists() and not any(gpu_dir.iterdir()):
                gpu_dir.rmdir()
                file_dir = gpu_dir.parent
                if file_dir.exists() and not any(file_dir.iterdir()):
                    file_dir.rmdir()

        return len(to_remove)

    def get_runs_for_file(
        self,
        source_filename: str,
        gpu_type: str,
    ) -> list[Path]:
        """Get all run directories for a specific file and GPU type.

        Args:
            source_filename: The source filename (e.g., "mandelbrot_gpu.cu").
            gpu_type: GPU type (e.g., "T4", "NONE").

        Returns:
            List of job directory paths, sorted by timestamp (newest first).
        """
        file_stem = Path(source_filename).stem
        gpu_dir = self.base_dir / file_stem / gpu_type

        if not gpu_dir.exists():
            return []

        job_dirs = [d for d in gpu_dir.iterdir() if d.is_dir()]
        return sorted(job_dirs, key=lambda p: p.name, reverse=True)

    def get_source_from_run(self, job_dir: Path, source_filename: str) -> str | None:
        """Get the source code from a specific run.

        Args:
            job_dir: Path to the job directory.
            source_filename: The source filename.

        Returns:
            Source code as string, or None if not found.
        """
        source_path = job_dir / source_filename
        if source_path.exists():
            return source_path.read_text()
        return None

    def diff_runs(
        self,
        source_filename: str,
        gpu_type: str,
        run_index: int = 1,
        context_lines: int = 3,
    ) -> tuple[str | None, Path | None, Path | None]:
        """Generate a diff between two runs.

        Args:
            source_filename: The source filename.
            gpu_type: GPU type.
            run_index: Which run to compare against (1 = previous, 2 = two runs ago, etc.)
            context_lines: Number of context lines in diff.

        Returns:
            Tuple of (diff_string, newer_run_path, older_run_path).
            diff_string is None if no diff or not enough runs.
        """
        runs = self.get_runs_for_file(source_filename, gpu_type)

        if len(runs) < run_index + 1:
            return None, None, None

        newer_run = runs[run_index - 1]  # 0 for index 1, etc.
        older_run = runs[run_index]

        newer_source = self.get_source_from_run(newer_run, source_filename)
        older_source = self.get_source_from_run(older_run, source_filename)

        if newer_source is None or older_source is None:
            return None, newer_run, older_run

        # Normalize whitespace for comparison (but keep original for display)
        newer_lines = newer_source.splitlines(keepends=True)
        older_lines = older_source.splitlines(keepends=True)

        # Check if files are identical (ignoring whitespace)
        newer_stripped = [line.strip() for line in newer_lines]
        older_stripped = [line.strip() for line in older_lines]

        if newer_stripped == older_stripped:
            return "", newer_run, older_run  # Empty string = no meaningful diff

        # Generate unified diff
        diff = difflib.unified_diff(
            older_lines,
            newer_lines,
            fromfile=f"{older_run.name}/{source_filename}",
            tofile=f"{newer_run.name}/{source_filename}",
            n=context_lines,
        )

        return "".join(diff), newer_run, older_run
