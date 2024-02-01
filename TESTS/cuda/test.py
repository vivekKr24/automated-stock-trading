import torch


def test_function(tensor):
    tensor = tensor * tensor
    print(tensor)


def test_fn_call():
    tensor = torch.tensor(2, device='cuda')
    test_function(tensor)


test_fn_call()
