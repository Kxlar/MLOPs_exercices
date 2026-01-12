import os
import pytest
import torch
from src.code_structure.data import corrupt_mnist, normalize

# On vérifie si les données existent pour éviter de faire planter la CI
DATA_EXISTS = os.path.exists("data/processed/train_images.pt")

@pytest.mark.skipif(not DATA_EXISTS, reason="Data files not found")
def test_data_loading():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    
    # Test des dimensions
    img, label = train[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= label <= 9

def test_normalize():
    img = torch.randn(1, 28, 28) * 10 + 5 # Moyenne 5, std 10
    norm_img = normalize(img)
    assert torch.isclose(norm_img.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(norm_img.std(), torch.tensor(1.0), atol=1e-5)