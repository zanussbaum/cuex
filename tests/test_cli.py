"""Tests for the CLI module."""

from typer.testing import CliRunner

from cuex.cli import app

runner = CliRunner()


def test_version():
    """Test the --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "cuex" in result.stdout


def test_no_args_shows_help():
    """Test that running without args shows help."""
    result = runner.invoke(app, [])
    # Typer returns exit code 2 when no_args_is_help=True
    assert result.exit_code == 2
    assert "Usage" in result.stdout


def test_invalid_file_extension(tmp_path):
    """Test that invalid file extensions are rejected."""
    # Create an actual file with invalid extension
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    result = runner.invoke(app, ["run", str(test_file)])
    assert result.exit_code == 1
    assert "Unsupported file type" in result.stdout
