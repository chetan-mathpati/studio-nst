import torch

from core.features import gram_matrix


def test_gram_matrix_shape():
    x = torch.randn(1, 8, 16, 16)
    gram = gram_matrix(x)
    assert gram.shape == (1, 8, 8)
