import pytest
import torch
import os
from typer.testing import CliRunner
from src.code_structure.evaluate import app

runner = CliRunner()


@pytest.mark.skipif(
    not os.path.exists("models/model.pth"), reason="Model checkpoint not found"
)
def test_evaluate_cli():
    result = runner.invoke(app, ["models/model.pth"])

    assert result.exit_code == 0
    assert "Test accuracy" in result.stdout
