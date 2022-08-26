import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def gen_rician_examples(data_shape, type, alpha):

    fwd_fading_stats = sio.loadmat("Channel_Files/Forward_Channel.mat")
    fwd_fading_stats = fwd_fading_stats["Channel"]
    fwd_fading_stats = torch.tensor(fwd_fading_stats)

    fdb_fading_stats = sio.loadmat("Channel_Files/Backward_Channel.mat")
    fdb_fading_stats = fdb_fading_stats["Channel"]
    fdb_fading_stats = torch.tensor(fdb_fading_stats)

    rand_subc = torch.randint(high=48, size=data_shape)

    K = fwd_fading_stats[rand_subc, 0]
    A = fwd_fading_stats[rand_subc, 1]
    fwd_fading = gen_fading(data_shape, type, K, A)

    K = fdb_fading_stats[rand_subc, 0]
    A = fdb_fading_stats[rand_subc, 1]
    fdb_fading = gen_fading(data_shape, type, K, A)

    # Assuming phase information at the receiver
    fwd_noise_stats = sio.loadmat("Channel_Files/Forward_Noise.mat")
    fwd_noise_stats = fwd_noise_stats["Noise"]
    fwd_noise_stats = torch.tensor(fwd_noise_stats)

    mean = fwd_noise_stats[rand_subc, 0]
    var = fwd_noise_stats[rand_subc, 1]
    fwd_noise = gen_noise(data_shape, mean, var)

    fdb_noise_stats = sio.loadmat("Channel_Files/Backward_Noise.mat")
    fdb_noise_stats = fdb_noise_stats["Noise"]
    fdb_noise_stats = torch.tensor(fwd_noise_stats)

    mean = fdb_noise_stats[rand_subc, 0]
    var = fdb_noise_stats[rand_subc, 1]
    fdb_noise = gen_noise(data_shape, mean, var)

    # Assuming phase information at the receiver
    return alpha * fwd_fading, fdb_fading, fwd_noise, fdb_noise


def gen_noise(data_shape, mean, var):
    Real = torch.randn(data_shape)
    Imag = torch.randn(data_shape)
    Noise = mean + torch.sqrt(var) * torch.complex(Real, Imag)
    return Noise


def gen_fading(data_shape, type, K, A):

    coeffLOS = torch.sqrt(K / (K + 1))
    coeffNLOS = torch.sqrt(1 / (K + 1))
    if type == "fast":
        hLOSReal = torch.ones(
            data_shape
        )  # Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn(data_shape)
        hNLOSImag = torch.randn(data_shape)
    else:  # Slow fading case
        hLOSReal = torch.ones(
            data_shape
        )  # Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn((data_shape[0], 1, 1)).repeat(
            1, data_shape[1], data_shape[2]
        )
        hNLOSImag = torch.randn((data_shape[0], 1, 1)).repeat(
            1, data_shape[1], data_shape[2]
        )
    fading_h = torch.abs(coeffLOS * torch.complex(hLOSReal, hLOSImag) + coeffNLOS * torch.complex(
        hNLOSReal, hNLOSImag)
    )
    return A * fading_h


# plt.savefig('example2.png')
