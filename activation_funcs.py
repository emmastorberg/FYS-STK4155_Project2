import autograd.numpy as np # type: ignore


def ReLU(z):
    return np.where(z > 0, z, 0)


def signmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]
