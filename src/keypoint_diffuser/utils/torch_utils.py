import torch


def no_grad(cls):
    for name, attr in cls.__dict__.items():
        if callable(attr) and not name.startswith("__"):
            setattr(cls, name, torch.no_grad()(attr))
    return cls
