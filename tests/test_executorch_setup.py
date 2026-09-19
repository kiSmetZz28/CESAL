"""Runtime reuse must not reinstall dependencies in an existing container."""
import sys
from unittest.mock import Mock

from tools import setup_executorch as setup


def test_existing_runner_without_bindings_needs_no_writes(tmp_path, monkeypatch, capsys):
    runner = tmp_path / "executor_runner"
    runner.write_bytes(b"existing runtime")
    runner.chmod(0o555)
    monkeypatch.setattr(setup, "EXECUTOR_RUNNER", runner)
    monkeypatch.setitem(sys.modules, "executorch.runtime", None)
    install = Mock(side_effect=AssertionError("Must not install optional bindings"))
    download = Mock(side_effect=AssertionError("Must not download installed runtime"))
    monkeypatch.setattr(setup, "install_python_bindings", install)
    monkeypatch.setattr(setup, "gdrive_download", download)
    before = runner.stat()

    assert setup.setup_executorch(prefix="[1/4] ")
    assert setup.setup_executorch()

    install.assert_not_called()
    download.assert_not_called()
    assert runner.read_bytes() == b"existing runtime"
    assert runner.stat().st_mtime_ns == before.st_mtime_ns
    output = capsys.readouterr().out
    assert "[1/4] ExecuTorch runtime already installed" in output
    assert "WARNING" not in output


def test_fresh_runtime_keeps_binding_install_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "executorch.runtime", None)
    monkeypatch.setattr(setup, "executorch_present", Mock(side_effect=[False, False, True]))
    monkeypatch.setattr(setup, "ensure_gdown", Mock())
    monkeypatch.setattr(setup, "gdrive_download", Mock(return_value=True))
    extract = Mock()
    install = Mock(return_value=False)
    monkeypatch.setattr(setup, "extract_zip", extract)
    monkeypatch.setattr(setup, "install_python_bindings", install)

    assert setup.setup_executorch()
    extract.assert_called_once()
    install.assert_called_once()


def test_missing_runtime_download_failure_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "executorch.runtime", None)
    monkeypatch.setattr(setup, "executorch_present", Mock(return_value=False))
    monkeypatch.setattr(setup, "ensure_gdown", Mock())
    monkeypatch.setattr(setup, "gdrive_download", Mock(return_value=False))
    install = Mock()
    monkeypatch.setattr(setup, "install_python_bindings", install)

    assert not setup.setup_executorch()
    install.assert_not_called()
