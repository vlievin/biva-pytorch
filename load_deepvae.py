import argparse
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from biva.utils.restore import restore_session
from booster.utils import logging_sep
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def build_and_save_grid(data, logdir, filename, N=100):
    nrow = math.floor(math.sqrt(N))
    grid = make_grid(data, nrow=nrow)

    # normalize
    grid -= grid.min()
    grid /= grid.max()

    # save the raw image
    img = grid.data.permute(1, 2, 0).cpu().numpy()
    matplotlib.image.imsave(os.path.join(logdir, f"{filename}.png"), img)

    plt.figure(figsize=(8, 8))
    plt.title(filename.split('/')[-1])
    plt.imshow(img);
    plt.axis('off')
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='runs/binmnist-biva-seed42', help='directory containing the training session')
parser.add_argument('--bs', default=36, type=int, help='batch size')
parser.add_argument('--device', default='auto', help='auto, cuda, cpu')
args = parser.parse_args()

session = restore_session(args.logdir, args.device)
model, likelihood, evaluator, device, run_id = [session[k] for k in
                                                ['model', 'likelihood', 'evaluator', 'device', 'run_id']]

logdir = os.path.join('output', run_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)

dataset = DataLoader(session['test_dataset'], batch_size=args.bs, shuffle=True, pin_memory=False, num_workers=0)

with torch.no_grad():
    x = next(iter(dataset)).to(device)
    build_and_save_grid(x, logdir, "original")

    # display posterior samples x ~ p(x|z), z ~ q(z|x)
    x_ = model(x).get('x_')
    x_ = likelihood(logits=x_).sample()
    build_and_save_grid(x_, logdir, "posterior")

    # dislay prior samples x ~ p(x|z), z ~ p(z)
    x_ = model.sample_from_prior(100).get('x_')
    x_ = likelihood(logits=x_).sample()
    build_and_save_grid(x_, logdir, "prior")

    print(logging_sep("="))
    print(f"Samples logged in {logdir}")

    # evaluate the likelihood on the batch of data
    _, diagnostics, _ = evaluator(model, x)
    print(logging_sep("="))
    print(diagnostics)
    print(logging_sep("="))
