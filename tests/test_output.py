"""Tests for the output module."""

import tempfile
from pathlib import Path

from cuex.output import OutputManager


def test_save_job_output():
    """Test saving job output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputManager(base_dir=Path(tmpdir))

        job_dir = manager.save_job_output(
            job_id="20241215_120000_abc12345",
            source_code='#include <stdio.h>\nint main() { printf("Hello"); }',
            source_filename="test.cpp",
            gpu_type="NONE",
            compile_log="Compiling...\n",
            execute_log="Hello",
        )

        assert job_dir.exists()
        # Check nested structure: <base>/test/NONE/<job_id>/
        assert job_dir == Path(tmpdir) / "test" / "NONE" / "20241215_120000_abc12345"
        assert (job_dir / "test.cpp").exists()
        assert (job_dir / "compile_log.txt").exists()
        assert (job_dir / "execute_log.txt").exists()


def test_save_job_with_asm():
    """Test saving job output with ASM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputManager(base_dir=Path(tmpdir))

        job_dir = manager.save_job_output(
            job_id="20241215_120000_abc12345",
            source_code="int main() {}",
            source_filename="test.cpp",
            gpu_type="NONE",
            compile_log="",
            execute_log="",
            asm_output=".text\nmain:\n  ret",
        )

        assert (job_dir / "asm_output.txt").exists()
        assert ".text" in (job_dir / "asm_output.txt").read_text()


def test_list_jobs():
    """Test listing job directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputManager(base_dir=Path(tmpdir))

        # Create some job directories across different files and GPUs
        for i in range(3):
            manager.save_job_output(
                job_id=f"20241215_12000{i}_abc{i}",
                source_code="",
                source_filename="test.cpp",
                gpu_type="T4" if i % 2 == 0 else "NONE",
                compile_log="",
                execute_log="",
            )

        jobs = manager.list_jobs()
        assert len(jobs) == 3


def test_cleanup_old_jobs():
    """Test cleaning up old job directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputManager(base_dir=Path(tmpdir))

        # Create 5 job directories
        for i in range(5):
            manager.save_job_output(
                job_id=f"20241215_12000{i}_abc{i}",
                source_code="",
                source_filename="test.cpp",
                gpu_type="T4",
                compile_log="",
                execute_log="",
            )

        # Keep only 2
        removed = manager.cleanup_old_jobs(keep=2)
        assert removed == 3

        jobs = manager.list_jobs()
        assert len(jobs) == 2
