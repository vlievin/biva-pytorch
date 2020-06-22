import json
import os
import pickle

from booster.utils import available_device
from torch.distributions import Bernoulli

from .discretized_mixture_logits import DiscretizedMixtureLogits
from .logging import load_model
from ..datasets import get_binmnist_datasets, get_cifar10_datasets
from ..evaluation import VariationalInference
from ..model.deepvae import DeepVae


def restore_session(logdir, device='auto'):
    """load model from a saved session"""

    if logdir[-1] == '/':
        logdir = logdir[:-1]
    run_id = logdir.split('/')[-1]

    # load the hyperparameters and arguments
    hyperparameters = pickle.load(open(os.path.join(logdir, "hyperparameters.p"), "rb"))
    opt = json.load(open(os.path.join(logdir, "config.json")))

    # instantiate the model
    model = DeepVae(**hyperparameters)
    device = available_device() if device == 'auto' else device
    model.to(device)

    # load pretrained weights
    load_model(model, logdir)

    # define likelihood and evaluator
    likelihood = {'cifar': DiscretizedMixtureLogits(opt['nr_mix']), 'binmnist': Bernoulli}[opt['dataset']]
    evaluator = VariationalInference(likelihood, iw_samples=1)

    # load the dataset
    if opt['dataset'] == 'binmnist':
        train_dataset, valid_dataset, test_dataset = get_binmnist_datasets(opt['data_root'])
    elif opt['dataset'] == 'cifar10':
        from torchvision.transforms import Lambda

        transform = Lambda(lambda x: x * 2 - 1)
        train_dataset, valid_dataset, test_dataset = get_cifar10_datasets(opt.data_root, transform=transform)
    else:
        raise NotImplementedError

    return {
        'model': model,
        'device': device,
        'run_id': run_id,
        'hyperparameters': hyperparameters,
        'opt': hyperparameters,
        'likelihood': likelihood,
        'evaluator': evaluator,
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
        'test_dataset': test_dataset,
    }
