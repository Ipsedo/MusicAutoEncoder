import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal


def rec_normal_gen_mask(hidden_size: int, nb_sec: int, nb_channel: int, eta: float = 0.8) -> th.Tensor:
    res = th.zeros(nb_sec * hidden_size, nb_channel)
    res[0] = th.randn(nb_channel) * th.randint(0, 2, (nb_channel,), dtype=th.float)

    for i in range(1, nb_sec * hidden_size):
        res[i] = eta * res[i - 1] + (1. - eta) * th.randn(nb_channel) * th.randint(0, 2, (nb_channel,), dtype=th.float)

    return res.permute(1, 0).unsqueeze(0)


def rec_normal_gen(hidden_size: int, nb_sec: int, nb_channel: int, eta: float = 0.8) -> th.Tensor:
    res = th.zeros(nb_sec * hidden_size, nb_channel)
    res[0] = th.randn(nb_channel)

    for i in range(1, nb_sec * hidden_size):
        res[i] = eta * res[i - 1] + (1. - eta) * th.randn(nb_channel)

    return res.permute(1, 0).unsqueeze(0)


def rec_multivariate_gen(hidden_size: int, nb_sec: int, nb_channel: int,
                         means: th.Tensor, cov_matrix: th.Tensor,
                         eta: float = 0.2) -> th.Tensor:
    assert len(means.size()) == 1, \
        f"Wrong mean size length, actual : {len(means.size())}, needed : {1}."
    assert len(cov_matrix.size()) == 2, \
        f"Wrong covariance matrix size length, actual {len(cov_matrix.size())}, needed : {2}"
    assert cov_matrix.size(0) == cov_matrix.size(1) == means.size(0) == nb_channel, \
        f"Wrong size equality in means or cov_mat or nb_channel, " \
        f"means : {means.size()} cov_mat : {cov_matrix.size()} nb_channel : {nb_channel}"

    dist = MultivariateNormal(means, cov_matrix)

    res = th.zeros(nb_sec * hidden_size, nb_channel)

    res[0] = dist.sample()

    for i in range(1, nb_sec * hidden_size):
        res[i] = eta * res[i - 1] + (1. - eta) * dist.sample()

    return res.to(th.float).transpose(1, 0).unsqueeze(0)


def gen_random_means(nb_channel: int, min_v: float, max_v: float) -> th.Tensor:
    return th.randn(nb_channel) * (max_v - min_v) + min_v


def gen_random_cov_mat(nb_channel: int, extreme_value: float) -> th.Tensor:
    cov_mat = th.randn(nb_channel, nb_channel) * extreme_value
    cov_mat = th.mm(cov_mat.transpose(1, 0), cov_mat)
    return cov_mat


def gen_init_normal_uni_add(hidden_size: int, nb_sec: int, nb_channel: int) -> th.Tensor:
    res = th.zeros(nb_sec * hidden_size, nb_channel)
    res[0] = th.randn(nb_channel) * th.randint(0, 2, (nb_channel,), dtype=th.float)

    for i in range(1, nb_sec * hidden_size):
        res[i] = res[i - 1] + th.rand(nb_channel) * 2e-2 - 1e-2

    return res.permute(1, 0).unsqueeze(0)


def rec_multivariate_different_gen(hidden_size: int, nb_sec: int, nb_channel: int,
                                   eta: float = 0.1, beta: float = 0.2) -> th.Tensor:
    means = gen_random_means(nb_channel, -0.3, 0.3)
    cov_mat = gen_random_cov_mat(nb_channel, 20.)
    dist = MultivariateNormal(means, cov_mat)
    res = th.zeros(nb_sec * hidden_size, nb_channel)

    mask = th.randint(0, 2, (nb_channel,)).to(th.float)

    res[0] = dist.sample() * mask

    for i in range(1, nb_sec * hidden_size):
        dist.loc = beta * dist.loc + (1. - beta) * gen_random_means(nb_channel, -0.3, 0.3)
        dist.covariance_matrix = beta * dist.covariance_matrix + (1. - beta) * gen_random_cov_mat(nb_channel, 20.)
        res[i] = eta * res[i - 1] + (1. - eta) * dist.sample()

        for _ in range(50):
            j = th.randint(0, nb_channel, (1,)).item()
            mask[j] = 1. - mask[j].item()

        res[i] *= mask

    return res.to(th.float).transpose(1, 0).unsqueeze(0)
