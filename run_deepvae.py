import argparse
import json
import logging
import os

import numpy as np
import torch
from biva.datasets import get_binmnist_datasets, get_cifar10_datasets
from biva.evaluation import VariationalInference
from biva.model import DeepVae, get_deep_vae_mnist, get_deep_vae_cifar, VaeStage, LvaeStage, BivaStage
from biva.utils import LowerBoundedExponentialLR, training_step, test_step, summary2logger, save_model, load_model, \
    sample_model, DiscretizedMixtureLogits
from booster.data import Aggregator
from booster.utils import EMA
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--data_root', default='data/', help='directory to store the dataset')
parser.add_argument('--dataset', default='binmnist', help='binmnist')
parser.add_argument('--model_type', default='biva', help='model type (vae | lvae | biva)')
parser.add_argument('--device', default='cuda', help='cuda or cpu')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--bs', default=48, type=int, help='batch size')
parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
parser.add_argument('--lr', default=2e-3, type=float, help='base learning rate')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--freebits', default=2.0, type=float, help='freebits per latent variable')
parser.add_argument('--nr_mix', default=10, type=int, help='number of mixtures')
parser.add_argument('--ema', default=0.9995, type=float, help='ema')
parser.add_argument('--q_dropout', default=0.5, type=float, help='inference model dropout')
parser.add_argument('--p_dropout', default=0.5, type=float, help='generative model dropout')
parser.add_argument('--iw_samples', default=1000, type=int, help='number of importance weighted samples for testing')
parser.add_argument('--id', default='', type=str, help='run id suffix')
parser.add_argument('--no_skip', action='store_true', help='do not use skip connections')
parser.add_argument('--log_var_act', default='softplus', type=str, help='activation for the log variance')

opt = parser.parse_args()

# set random seed, set run-id, init log directory and save config
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
run_id = f"{opt.dataset}-{opt.model_type}-seed{opt.seed}"
if len(opt.id):
    run_id += f"-{opt.id}"
if opt.log_var_act != 'none':
    run_id += f"-{opt.log_var_act}"
logdir = os.path.join(opt.root, run_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)
with open(os.path.join(logdir, 'config.json'), 'w') as fp:
    fp.write(json.dumps(vars(opt)))

# define tensorboard writers
train_writer = SummaryWriter(os.path.join(logdir, 'train'))
valid_writer = SummaryWriter(os.path.join(logdir, 'valid'))

# load data
if opt.dataset == 'binmnist':
    train_dataset, valid_dataset, test_dataset = get_binmnist_datasets(opt.data_root)
elif opt.dataset == 'cifar10':
    from torchvision.transforms import Lambda

    transform = Lambda(lambda x: x * 2 - 1)
    train_dataset, valid_dataset, test_dataset = get_cifar10_datasets(opt.data_root, transform=transform)
else:
    raise NotImplementedError

train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, pin_memory=False, num_workers=opt.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=2 * opt.bs, shuffle=True, pin_memory=False,
                          num_workers=opt.num_workers)
test_loader = DataLoader(test_dataset, batch_size=2 * opt.bs, shuffle=True, pin_memory=False,
                         num_workers=opt.num_workers)
tensor_shp = (-1, *train_dataset[0].shape)

# define likelihood
if 'cifar' in opt.dataset:
    likelihood = DiscretizedMixtureLogits(opt.nr_mix)
else:
    likelihood = Bernoulli

# define model
if 'cifar' in opt.dataset:
    stages, latents = get_deep_vae_cifar()
    features_out = 10 * opt.nr_mix
else:
    stages, latents = get_deep_vae_mnist()
    features_out = tensor_shp[1]

Stage = {'vae': VaeStage, 'lvae': LvaeStage, 'biva': BivaStage}[opt.model_type]
log_var_act = {'none': None, 'softplus': torch.nn.Softplus, 'tanh': torch.nn.Tanh}[opt.log_var_act]
model = DeepVae(Stage=Stage,
                tensor_shp=tensor_shp,
                stages=stages,
                latents=latents,
                nonlinearity='elu',
                q_dropout=opt.q_dropout,
                p_dropout=opt.p_dropout,
                type=opt.model_type,
                features_out=features_out,
                no_skip=opt.no_skip,
                log_var_act=log_var_act)

model.to(opt.device)

# define the evaluator
evaluator = VariationalInference(likelihood, iw_samples=1)

# define evaluation model with Exponential Moving Average
ema = EMA(model, opt.ema)

# data dependent init for weight normalization (automatically done during the first forward pass)
with torch.no_grad():
    x = next(iter(train_loader)).to(opt.device)
    model(x)

# print stages
print("########################### architecture:")
for i, (convs, z) in enumerate(zip(stages, latents)):
    print("# Stage =", i)
    print("Deterministic block:", convs)
    print("Stochastic layer:", z)

# define freebits
n_latents = len(latents)
if opt.model_type == 'biva':
    n_latents = 2 * n_latents - 1
freebits = [opt.freebits] * n_latents

# optimizer
optimizer = torch.optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999,))
scheduler = LowerBoundedExponentialLR(optimizer, 0.999999, 0.0001)

# logging utils
kwargs = {'beta': 1.0, 'freebits': freebits}
best_elbo = (-1e20, 0, 0)
global_step = 1
train_agg = Aggregator()
val_agg = Aggregator()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                              logging.StreamHandler()])
train_logger = logging.getLogger('train')
eval_logger = logging.getLogger('eval')
M_parameters = (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
logging.getLogger(run_id).info(f'# Total Number of Parameters: {M_parameters:.3f}M')

# init sample
sample_model(ema.model, likelihood, logdir, writer=valid_writer, global_step=global_step, N=100)

# run
for epoch in range(1, opt.epochs + 1):

    # training
    train_agg.initialize()
    for x in tqdm(train_loader, desc='train epoch'):
        x = x.to(opt.device)
        diagnostics = training_step(x, model, evaluator, optimizer, scheduler, **kwargs)
        train_agg.update(diagnostics)
        ema.update()
        global_step += 1
    train_summary = train_agg.data.to('cpu')

    # evaluation
    val_agg.initialize()
    for x in tqdm(valid_loader, desc='valid epoch'):
        x = x.to(opt.device)
        diagnostics = test_step(x, ema.model, evaluator, **kwargs)
        val_agg.update(diagnostics)
    eval_summary = val_agg.data.to('cpu')

    # keep best model
    best_elbo = save_model(ema.model, eval_summary, global_step, epoch, best_elbo, logdir)

    # logging
    summary2logger(train_logger, train_summary, global_step, epoch)
    summary2logger(eval_logger, eval_summary, global_step, epoch, best_elbo)

    # tensorboard logging
    train_summary.log(train_writer, global_step)
    eval_summary.log(valid_writer, global_step)

    # sample model
    sample_model(ema.model, likelihood, logdir, writer=valid_writer, global_step=global_step, N=100)

# load best model
load_model(ema.model, logdir)

# sample model
sample_model(ema.model, likelihood, logdir, N=100)

# final test
iw_evaluator = VariationalInference(likelihood, iw_samples=opt.iw_samples)
test_agg = Aggregator()
test_logger = logging.getLogger('test')
test_logger.info(f"best elbo at step {best_elbo[1]}, epoch {best_elbo[2]}: {best_elbo[0]:.3f} nats")

test_agg.initialize()
for x in tqdm(test_loader, desc='iw test epoch'):
    x = x.to(opt.device)
    diagnostics = test_step(x, ema.model, iw_evaluator, **kwargs)
    test_agg.update(diagnostics)
test_summary = test_agg.data.to('cpu')

summary2logger(test_logger, test_summary, best_elbo[1], best_elbo[2], None)
