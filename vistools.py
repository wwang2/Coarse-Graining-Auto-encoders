import torch
import numpy as np

from se3cnn.util.plot import spherical_harmonics_coeff_to_sphere

import plotly.graph_objects as go


def sh_coeff_to_xyz_signal(sh_coeff, angular_resolution, r_scale=1.0):
    sh_coeff = torch.from_numpy(sh_coeff)
    a = torch.linspace(0, 2 * np.pi, angular_resolution, dtype=sh_coeff.dtype)
    b = torch.linspace(0, np.pi, angular_resolution, dtype=sh_coeff.dtype)
    a, b = torch.meshgrid([a, b])

    signal = spherical_harmonics_coeff_to_sphere(sh_coeff, a, b)
    r = signal.relu() * r_scale
    x = r * a.cos() * b.sin()
    y = r * a.sin() * b.sin()
    z = r * b.cos()
    return x, y, z, signal


def xyz_signal_to_surface(signal_xyz, center, opacity=1.0):
    x, y, z, signal = signal_xyz
    xd, yd, zd = center
    x += xd
    y += yd
    z += zd
    return go.Surface(x=x, y=y, z=z, showscale=False, surfacecolor=signal, opacity=opacity)


def assignment_to_color(onehot, color_map):
    save_size = onehot.size()
    onehot = onehot.reshape(-1, save_size[-1])
    argmax = torch.argmax(onehot, -1)
    colors = np.array([color_map[i.item()] for i in argmax])
    colors = colors.reshape(*save_size[:-1], -1)
    return colors
