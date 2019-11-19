from time import time

import torch


def append_ellapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_ellapsed_time
def training_step(x, model, evaluator, optimizer, scheduler, **kwargs):
    optimizer.zero_grad()
    model.train()

    loss, diagnostics = evaluator(model, x, **kwargs)
    loss = loss.mean(0)

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return diagnostics


@torch.no_grad()
@append_ellapsed_time
def test_step(x, model, evaluator, **kwargs):
    model.eval()

    loss, diagnostics = evaluator(model, x, **kwargs)

    return diagnostics
