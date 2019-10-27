import math
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def summary2logger(logger, summary, global_step, epoch, best=None, stats_key='loss'):
    """write summary to logging"""
    if not stats_key in summary.keys():
        logger.warning('key ' + str(stats_key) + ' not int output dictionary')
    else:
        message = f'\t[{global_step} / {epoch}]   '
        message += ''.join([f'{k} {v:6.2f}   ' for k, v in summary.get(stats_key).items()])
        message += f'({summary["info"]["elapsed-time"]:.2f}s /iter)'
        if best is not None:
            message += f'   (best: {best[0]:6.2f}  [{best[1]} / {best[2]}])'
        logger.info(message)


def save_model(model, eval_summary, global_step, epoch, best_elbo, logdir):
    elbo = eval_summary['loss']['elbo']
    prev_elbo, *_ = best_elbo
    if elbo > prev_elbo:
        best_elbo = (elbo, global_step, epoch)
        pth = os.path.join(logdir, "model.pth")
        torch.save(model.state_dict(), pth)

    return best_elbo


def load_model(model, logdir):
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth")))


@torch.no_grad()
def sample_model(model, likelihood, logdir, N=100):
    # sample model
    x_ = model.sample_from_prior(N).get('x_')
    x_ = likelihood(logits=x_).sample()

    # make grid
    nrow = math.floor(math.sqrt(N))
    grid = make_grid(x_, nrow=nrow)

    # plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.data.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(logdir, "samples.png"))
    plt.close()
