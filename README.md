# BIVA (PyTorch)

Official PyTorch BIVA implementation (BIVA: A Very Deep Hierarchy of Latent Variables forGenerative Modeling) for binarized MNIST and CIFAR. The original Tensorflow implementation can be found [here](https://github.com/larsmaaloee/BIVA).

## run the experiments

```bash
conda create --name biva python=3.7
conda activate biva
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python run_deepvae.py --dataset binmnist --q_dropout 0.5 --p_dropout 0.5 --device cuda
CUDA_VISIBLE_DEVICES=0 python run_deepvae.py --dataset cifar10 --q_dropout 0.2 --p_dropout 0 --device cuda
```

## Citation

```
@article{maale2019biva,
    title={BIVA: A Very Deep Hierarchy of Latent Variables for Generative Modeling},
    author={Lars Maaløe and Marco Fraccaro and Valentin Liévin and Ole Winther},
    year={2019},
    eprint={1902.02102},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

## Pip package

### install requirements

* `pytorch 1.3.0`
* `torchvision`
* `matplotlib`
* `tensorboard`
* `booster-pytorch==0.0.2`

### install package

```bash
pip install git+https://github.com/vlievin/biva-pytorch.git
```

### build deep VAEs

```python
import torch
from torch.distributions import Bernoulli

from biva import DenseNormal, ConvNormal
from biva import VAE, LVAE, BIVA

# build a 2 layers VAE for binary images

# define the stochastic layers
z = [
    {'N': 8, 'kernel': 5, 'block': ConvNormal},  # z1
    {'N': 16, 'block': DenseNormal}  # z2
]

# define the intermediate layers
# each stage defines the configuration of the blocks for q_(z_{l} | z_{l-1}) and p_(z_{l-1} | z_{l})
# each stage is defined by a sequence of 3 resnet blocks
# each block is degined by a tuple [filters, kernel, stride]
stages = [
    [[64, 3, 1], [64, 3, 1], [64, 3, 2]],
    [[64, 3, 1], [64, 3, 1], [64, 3, 2]]
]

# build the model
model = VAE(tensor_shp=(-1, 1, 28, 28), stages=stages, latents=z, dropout=0.5)

# forward pass and data-dependent initialization
x = torch.empty((8, 1, 28, 28)).uniform_().bernoulli()
data = model(x)  # data = {'x_' : p(x|z), z \sim q(z|x), 'kl': [kl_z1, kl_z2]}

# sample from prior
data = model.sample_from_prior(N=16)  # data = {'x_' : p(x|z), z \sim p(z)}
samples = Bernoulli(logits=data['x_']).sample()

```
