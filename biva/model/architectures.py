from .stochastic import DenseNormal


def get_deep_vae_mnist():
    """
    Get the binary images Deep VAE configuration.
    :return: enc, z
    """
    filters = 64
    no_layers = 2
    enc = []
    z = []

    enc_z1 = [[filters, 5, 1]] * no_layers
    enc_z1 += [[filters, 5, 2]]
    z1 = {'N': 48, 'block': DenseNormal}
    enc += [enc_z1]
    z += [z1]

    enc_z2 = [[filters, 3, 1]] * no_layers
    enc_z2 += [[filters, 3, 1]]
    z2 = {'N': 40, 'block': DenseNormal}
    enc += [enc_z2]
    z += [z2]

    enc_z3 = [[filters, 3, 1]] * no_layers
    enc_z3 += [[filters, 3, 1]]
    z3 = {'N': 32, 'block': DenseNormal}
    enc += [enc_z3]
    z += [z3]

    enc_z4 = [[filters, 3, 1]] * no_layers
    enc_z4 += [[filters, 3, 1]]
    z4 = {'N': 24, 'block': DenseNormal}
    enc += [enc_z4]
    z += [z4]

    enc_z5 = [[filters, 3, 1]] * no_layers
    enc_z5 += [[filters, 3, 1]]
    z5 = {'N': 16, 'block': DenseNormal}
    enc += [enc_z5]
    z += [z5]

    enc_z6 = [[filters, 3, 1]] * no_layers
    enc_z6 += [[filters, 3, 2]]
    z6 = {'N': 8, 'block': DenseNormal}
    enc += [enc_z6]
    z += [z6]

    return enc, z
