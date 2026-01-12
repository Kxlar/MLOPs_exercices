from typer.testing import CliRunner
from src.code_structure.train import app
import os

runner = CliRunner()


def test_train_cli(tmp_path):
    # On crée un dossier temporaire pour les modèles pour ne pas écraser l'existant
    if not os.path.exists("models"):
        os.makedirs("models")

    # On lance l'entraînement sur 1 seule époque avec 1 seul batch pour aller vite
    result = runner.invoke(
        app, ["--lr", "1e-3", "--batch-size", "2", "--epochs", "1"]
    )

    assert result.exit_code == 0
    assert "Training complete" in result.stdout
    assert os.path.exists("models/model.pth")
