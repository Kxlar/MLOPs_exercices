from src.code_structure.model import MyAwesomeModel
import torch
import pytest
import re


def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    
    # Utilise re.escape pour que les [ ] soient traités comme du texte
    expected_msg = re.escape("Expected each sample to have shape [1, 28, 28]")
    with pytest.raises(ValueError, match=expected_msg):
        model(torch.randn(1, 1, 28, 29))
