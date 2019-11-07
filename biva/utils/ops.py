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
def training_step(x, pipeline, optimizer, scheduler, **kwargs):
    optimizer.zero_grad()
    pipeline.train()

    loss, diagnostics = pipeline(x, **kwargs)
    loss = loss.mean(0)

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return diagnostics


@torch.no_grad()
@append_ellapsed_time
def test_step(x, pipeline, **kwargs):
    pipeline.eval()

    loss, diagnostics = pipeline(x, **kwargs)

    return diagnostics
