import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from functools import partial

from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel
from se3cnn.non_linearities import rescaled_act


ACTS = {
    "sigmoid": rescaled_act.sigmoid,
    "tanh": rescaled_act.tanh,
    "relu": rescaled_act.relu,
    "absolute": rescaled_act.absolute,
    "softplus": rescaled_act.Softplus,
    "shifted_softplus": rescaled_act.ShiftedSoftplus,
}


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        radial_model = partial(
            CosineBasisModel,
            max_radius=args.rad_maxr,
            number_of_basis=args.rad_nb,
            h=args.rad_h,
            L=args.rad_L,
            act=ACTS[args.rad_act],
        )
        K = partial(Kernel, RadialModel=radial_model)
        C = partial(Convolution, K)

        if args.high_l_encoder:
            Rs = [
                (args.enc_l0, 0),
                (args.l1, 1),
                (args.l2, 2),
                (args.l3, 3),
                (args.l4, 4),
                (args.l5, 5),
            ]
            Rs = [[(args.atomic_nums, 0)]] + [Rs] * args.enc_L + [[(args.ncg, 0)]]
        else:
            Rs = [(args.enc_l0, 0)]
            Rs = [[(args.atomic_nums, 0)]] + [Rs] * args.enc_L + [[(args.ncg, 0)]]

        self.layers = torch.nn.ModuleList(
            [
                GatedBlock(Rs_in, Rs_out, ACTS[args.scalar_act], ACTS[args.gate_act], C)
                for Rs_in, Rs_out in zip(Rs, Rs[1:])
            ]
        )
        self.Rs = Rs

    def forward(self, features, geometry):
        output = features
        for layer in self.layers:
            output = layer(
                output.div(geometry.size(1) ** 0.5), geometry
            )  # Normalization of layers by number of atoms.
        return output


class Decoder(torch.nn.Module):
    def __init__(self, args, Rs_in=None, Rs_out=None):
        super().__init__()
        radial_model = partial(
            CosineBasisModel,
            max_radius=args.rad_maxr,
            number_of_basis=args.rad_nb,
            h=args.rad_h,
            L=args.rad_L,
            act=ACTS[args.rad_act],
        )
        K = partial(Kernel, RadialModel=radial_model)
        C = partial(Convolution, K)

        if Rs_in is None:
            if args.cg_ones:
                Rs_in = [[(1, 0)]]
            elif args.cg_specific_atom:
                Rs_in = [[(1, 0), (1, 2)]]
            else:
                Rs_in = [[(args.ncg, 0)]]

        Rs_middle = [
            (args.l0, 0),
            (args.l1, 1),
            (args.l2, 2),
            (args.l3, 3),
            (args.l4, 4),
            (args.l5, 5),
            (5, 6),
            # (5, 7)
        ]

        if Rs_out is None:
            Rs_out = [
                [(mul, l) for l, mul in enumerate([1] * (args.proj_lmax + 1))]
                * args.atomic_nums
            ]

        Rs = Rs_in + [Rs_middle] * args.dec_L + Rs_out

        self.layers = torch.nn.ModuleList(
            [
                GatedBlock(Rs_in, Rs_out, ACTS[args.scalar_act], ACTS[args.gate_act], C)
                for Rs_in, Rs_out in zip(Rs, Rs[1:])
            ]
        )
        self.Rs = Rs

    def forward(self, features, geometry):
        output = features
        for layer in self.layers:
            output = layer(
                output.div(geometry.size(1) ** 0.5), geometry
            )  # Normalization of layers by number of atoms.
        return output


def nearest_assignment(cg, atoms, dim=-1, dtype=None):
    assert len(cg.shape) == len(atoms.shape)
    assert cg.device == atoms.device
    device = cg.device

    if len(cg.shape) == 2:
        batch_mod = -1
    elif len(cg.shape) == 3:
        batch_mod = 0
    else:
        raise ValueError
    assign_index = (
        (atoms.unsqueeze(1 + batch_mod) - cg.unsqueeze(2 + batch_mod))
        .norm(dim=dim)
        .argmin(1 + batch_mod)
    )
    assign = torch.zeros(atoms.shape, dtype=dtype, device=device).scatter_(
        dim, assign_index.unsqueeze(dim), 1.0
    )
    return assign
