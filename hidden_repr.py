import torch as th

import numpy as np
import values


def multivariate_gen(n_sample: int, n_channel: int, mu: float = 0, sigma: float = 0.5) -> np.ndarray:
    means = np.zeros((n_channel,)) + mu
    cov_m = np.diag(np.ones((n_channel,)) * sigma)
    return np.random.multivariate_normal(means, cov_m, n_sample)


def multivariate_random_gen(n_sample: int, n_channel: int) -> np.ndarray:
    means = np.random.rand(n_channel) * 2. - 1.
    cov_m = np.diag(np.random.rand(n_channel))
    return np.random.multivariate_normal(means, cov_m, (n_sample, values.HIDDEN_LENGTH)).transpose((0, 2, 1))


def rec_normal_gen(hidden_size: int, nb_sec: int, nb_channel: int, eta: float = 0.8) -> th.Tensor:
    res = th.zeros(nb_sec * hidden_size, nb_channel)
    res[0] = th.randn(nb_channel) * th.randint(0, 2, (nb_channel,), dtype=th.float)

    for i in range(1, nb_sec * hidden_size):
        res[i] = eta * res[i - 1] + (1. - eta) * th.randn(nb_channel) * th.randint(0, 2, (nb_channel,), dtype=th.float)

    return res.permute(1, 0).unsqueeze(0)


def rec_multivariate_gen(n_sample: int, n_channel: int,
                         eta: float = 0.9, alpha: float = 0.8, seed: int = 1234) -> np.ndarray:
    np.random.seed(seed)

    res = np.zeros((n_sample * values.HIDDEN_LENGTH, n_channel)).astype(np.float16)

    means = np.random.rand(n_channel) * 2. - 1.
    cov_m = np.diag(np.random.rand(n_channel))

    random_mask = np.random.randint(0, 2, n_channel)

    res[0] = np.random.multivariate_normal(means, cov_m, 1) * random_mask

    for i in range(1, n_sample * values.HIDDEN_LENGTH):
        means = means * alpha + (np.random.rand(n_channel) * 2. - 1.) * (1. - alpha)
        cov_m = cov_m * alpha + np.diag(np.random.rand(n_channel)) * (1. - alpha)
        random_mask = np.random.randint(0, 2, n_channel)
        res[i] = (res[i - 1] * eta + np.random.multivariate_normal(means, cov_m, 1) * (1. - eta)) * random_mask

    return res.transpose((1, 0))


def rec_multivariate_gen_2(n_sample: int, n_channel: int,
                           n_change: int = 200, n_change_param: int = 200, seed: int = 1234) -> np.ndarray:
    np.random.seed(seed)

    res = np.zeros((n_sample * values.HIDDEN_LENGTH, n_channel)).astype(np.float16)

    means = np.random.rand(n_channel) * 2. - 1.
    cov_m = np.diag(np.random.rand(n_channel))

    res[0] = np.random.multivariate_normal(means, cov_m, 1)

    for i in range(1, n_sample * values.HIDDEN_LENGTH):
        alpha = np.ones((n_channel,))
        random_idx = np.random.randint(0, n_channel, n_change_param)
        alpha[random_idx] = 0

        means = means * alpha + (np.random.rand(n_channel) * 2. - 1.) * (1. - alpha)
        cov_m = np.diag(np.diag(cov_m) * alpha + np.random.rand(n_channel) * (1. - alpha))

        eta = np.ones((n_channel,))
        random_idx = np.random.randint(0, n_channel, n_change)
        eta[random_idx] = 0

        res[i] = res[i - 1] * eta + np.random.multivariate_normal(means, cov_m, 1) * (1. - eta)

    return res.transpose((1, 0))
