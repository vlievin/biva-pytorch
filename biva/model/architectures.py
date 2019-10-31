from .stochastic import DenseNormal, ConvNormal


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


def get_deep_vae_cifar():
    filters = 96
    no_layers = 2
    enc = []
    z = []

    enc_z1 = [[filters, 5, 1]] * no_layers
    enc_z1 += [[filters, 5, 2]]
    z_1 = {'N': 38, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z1]
    z += [z_1]

    enc_z2 = [[filters, 3, 1]] * no_layers
    enc_z2 += [[filters, 3, 1]]
    z_2 = {'N': 36, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z2]
    z += [z_2]

    enc_z3 = [[filters, 3, 1]] * no_layers
    enc_z3 += [[filters, 3, 1]]
    z_3 = {'N': 34, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z3]
    z += [z_3]

    enc_z4 = [[filters, 3, 1]] * no_layers
    enc_z4 += [[filters, 3, 1]]
    z_4 = {'N': 32, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z4]
    z += [z_4]

    enc_z5 = [[filters, 3, 1]] * no_layers
    enc_z5 += [[filters, 3, 1]]
    z_5 = {'N': 30, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z5]
    z += [z_5]

    enc_z6 = [[filters, 3, 1]] * no_layers
    enc_z6 += [[filters, 3, 1]]
    z_6 = {'N': 28, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z6]
    z += [z_6]

    enc_z7 = [[filters, 3, 1]] * no_layers
    enc_z7 += [[filters, 3, 1]]
    z_7 = {'N': 26, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z7]
    z += [z_7]

    enc_z8 = [[filters, 3, 1]] * no_layers
    enc_z8 += [[filters, 3, 1]]
    z_8 = {'N': 24, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z8]
    z += [z_8]

    enc_z9 = [[filters, 3, 1]] * no_layers
    enc_z9 += [[filters, 3, 1]]
    z_9 = {'N': 22, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z9]
    z += [z_9]

    enc_z10 = [[filters, 3, 1]] * no_layers
    enc_z10 += [[filters, 3, 1]]
    z_10 = {'N': 20, 'kernel': 16, 'block': ConvNormal}
    enc += [enc_z10]
    z += [z_10]

    enc_z11 = [[filters, 3, 1]] * no_layers
    enc_z11 += [[filters, 3, 2]]
    z_11 = {'N': 18, 'kernel': 8, 'block': ConvNormal}
    enc += [enc_z11]
    z += [z_11]

    enc_z12 = [[filters, 3, 1]] * no_layers
    enc_z12 += [[filters, 3, 1]]
    z_12 = {'N': 16, 'kernel': 8, 'block': ConvNormal}
    enc += [enc_z12]
    z += [z_12]

    enc_z13 = [[filters, 3, 1]] * no_layers
    enc_z13 += [[filters, 3, 1]]
    z_13 = {'N': 14, 'kernel': 8, 'block': ConvNormal}
    enc += [enc_z13]
    z += [z_13]

    enc_z14 = [[filters, 3, 1]] * no_layers
    enc_z14 += [[filters, 3, 1]]
    z_14 = {'N': 12, 'kernel': 8, 'block': ConvNormal}
    enc += [enc_z14]
    z += [z_14]

    enc_z15 = [[filters, 3, 1]] * no_layers
    enc_z15 += [[filters, 3, 2]]
    z_15 = {'N': 10, 'kernel': 4, 'block': ConvNormal}
    enc += [enc_z15]
    z += [z_15]

    return enc, z
