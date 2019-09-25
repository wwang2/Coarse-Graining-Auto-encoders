import torch
import torch.nn.functional

import mdtraj as md
import numpy as np


def temp_scheduler(epochs, decay_rate, t0, tmin, dtype=None, device=None):
    if decay_rate is None:
        schedule = np.linspace(0, epochs, epochs)
        decay_rate = (np.log(t0) - np.log(tmin)) / epochs
        schedule = t0 * np.exp(-schedule * decay_rate)
    else:
        schedule = np.linspace(0, epochs, epochs)
        decay_epoch = int(epochs * decay_rate)
        schedule = t0 * np.exp(-schedule / decay_epoch) + tmin
    return torch.from_numpy(schedule).to(dtype=dtype, device=device)


def otp_element_to_onehot():
    return {"H": [1, 0], "C": [0, 1]}


def load_data(num_training_examples):
    otp = md.load("data/otp.pdb")
    otp_top = otp.top.to_dataframe()[0]
    otp_element = otp_top["element"].values.tolist()
    xyz = np.load("data/otp_xyz.npy")
    force = np.load("data/otp_force.npy")
    xyz = xyz[-num_training_examples:] * 10
    force = force[-num_training_examples:] * 0.0239
    return xyz, force, otp_element
