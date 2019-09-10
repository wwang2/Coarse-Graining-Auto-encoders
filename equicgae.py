import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

import numpy as np
import mdtraj as md
import argparse
from functools import partial
from time import perf_counter
from tqdm import tqdm

from se3cnn.non_linearities import rescaled_act
from se3cnn.non_linearities.gated_block import GatedBlock
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import Convolution
from se3cnn.point.radial import CosineBasisModel
from se3cnn.SO3 import spherical_harmonics_xyz

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--nte", type=int, default=1000, help="Number of test examples.")
parser.add_argument("--ntr", type=int, default=3000, help="Number of training examples.")

# Task specific hyper parameters
parser.add_argument("--atomic_nums", type=int, default=2, help="Count of possible atomic numbers.")
parser.add_argument("--ncg", type=int, default=3, help="Count of coarse grained united atoms.")
parser.add_argument("--temp", type=float, default=4.0, help="Set initial temperature.")
parser.add_argument("--tmin", type=float, default=0.2, help="Set minimum temperature.")
parser.add_argument("--tdr", type=float, default=0.4, help="Temperature decay rate.")
parser.add_argument("--fm_epoch", type=int, default=400, help="Which epoch should force matching being.")
parser.add_argument("--fm_co", type=float, default=0.005, help="Coefficient multiplied by force matching loss.")
parser.add_argument("--scalar_encoder", action='store_true', help="Set the encoder to only deal in scalar values.")

# General hyper parameters
ACTS = ['sigmoid', 'tanh', 'relu', 'absolute', 'softplus', 'shifted_softplus']
parser.add_argument("--scalar_act", type=str, default="relu", choices=ACTS, help="Select scalar activation.")
parser.add_argument("--gate_act", type=str, default="sigmoid", choices=ACTS, help="Select gate activation.")
parser.add_argument("--softplus_beta", type=float, default=1.0, help="Which beta for softplus and shifted softplus?")
parser.add_argument("--bs", type=int, default=32, help="Batch size.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--l0", type=int, default=5)
parser.add_argument("--enc_l0", type=int, default=5, help="l0 multiplicity for the encoder.")
parser.add_argument("--l1", type=int, default=5)
parser.add_argument("--l2", type=int, default=5)
parser.add_argument("--l3", type=int, default=5)
parser.add_argument("--l4", type=int, default=5)
parser.add_argument("--l5", type=int, default=5)
parser.add_argument("--enc_L", type=int, default=3, help="How many layers to create for the encoder.")
parser.add_argument("--dec_L", type=int, default=1, help="How many layers to create for the decoder.")
parser.add_argument("--rad_act", type=str, default="relu", choices=ACTS, help="Select radial activation.")
parser.add_argument("--rad_nb", type=int, default=20, help="Radial number of bases.")
parser.add_argument("--rad_maxr", type=float, default=3, help="Max radius.")
parser.add_argument("--rad_h", type=int, default=50, help="Size of radial weight parameters.")
parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
parser.add_argument("--proj_lmax", type=int, default=5, help="What is the l max for projection.")

# Calculation
parser.add_argument("-e", "--experiment", action='store_true', help="Run experiment function.")
parser.add_argument("--cpu", action='store_true', help="Force calculation on cpu.")
parser.add_argument("--gpu", type=int, default=0, choices=list(range(torch.cuda.device_count())), help="Which gpu?")
parser.add_argument("--double", action='store_true', help="Calculate using double precision.")
parser.add_argument("--wall", type=float, default=5 * 60, help="If calc time is too long, break.")
parser.add_argument("--epochs", type=int, default=800, help="Number of epochs to calculate.")

# Saving
parser.add_argument("--pickle", type=str, default='out.pkl', help="File for results.")
parser.add_argument("--save_state", action='store_true', help="Save encoder and decoder state. Default False.")

group = parser.add_mutually_exclusive_group()
args = parser.parse_args()

# Global variables
if args.double:
    PRECISION = torch.float64
else:
    PRECISION = torch.float32
torch.set_default_dtype(PRECISION)

if torch.cuda.is_available() and not args.cpu:
    DEVICE = torch.device(f"cuda:{args.gpu}")
else:
    DEVICE = torch.device("cpu")
print(f"Calculating on {DEVICE}.")

ACT_FNS = [rescaled_act.sigmoid, rescaled_act.tanh, rescaled_act.relu, rescaled_act.absolute,
           rescaled_act.Softplus(args.softplus_beta), rescaled_act.ShiftedSoftplus(args.softplus_beta)]
ACTS = {act: fn for act, fn in zip(ACTS, ACT_FNS)}


def load_data(args):
    otp = md.load("data/otp.pdb")
    otp_top = otp.top.to_dataframe()[0]
    otp_element = otp_top['element'].values.tolist()
    xyz = np.load('data/otp_xyz.npy')
    force = np.load('data/otp_force.npy')
    xyz = xyz[-args.ntr:] * 10
    force = force[-args.ntr:] * 0.0239
    return xyz, force, otp_element


def project_to_ylm(relative_coords, l_max=5, dtype=PRECISION, device=DEVICE):
    sh = spherical_harmonics_xyz(range(l_max + 1), relative_coords, dtype=dtype, device=device)
    rank = len(sh.shape)
    return sh.permute(*range(1, rank), 0)


def sample_gumbel(shape, dtype=PRECISION, device=DEVICE, eps=1e-10):
    uniform = torch.rand(shape, dtype=dtype, device=device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dtype=PRECISION, device=DEVICE):
    y = logits + sample_gumbel(logits.size(), dtype=dtype, device=device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, dtype=PRECISION, device=DEVICE):
    """Sample a gumbel softmax distribution in dim = -1. Returns gumble, straight-through gumble."""
    y = gumbel_softmax_sample(logits, temperature, dtype=dtype, device=device)
    y_hard = torch.zeros(y.shape, dtype=dtype, device=device).scatter_(-1, y.argmax(-1).unsqueeze(-1), 1.0)
    y_hard = (y_hard - y).detach() + y
    return y, y_hard
    # if hard:
    #     y_hard = torch.zeros(y.shape, dtype=dtype, device=device).scatter_(-1, y.argmax(-1).unsqueeze(-1), 1.0)
    #     y = (y_hard - y).detach() + y
    # return y


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        radial_model = partial(
            CosineBasisModel,
            max_radius=args.rad_maxr,
            number_of_basis=args.rad_nb,
            h=args.rad_h,
            L=args.rad_L,
            act=ACTS[args.rad_act]
        )
        K = partial(Kernel, RadialModel=radial_model)
        C = partial(Convolution, K)

        if args.scalar_encoder:
            Rs = [(args.enc_l0, 0)]
            Rs = [[(args.atomic_nums, 0)]] + [Rs] * args.enc_L + [[(args.ncg, 0)]]
        else:
            Rs = [(args.enc_l0, 0), (args.l1, 1), (args.l2, 2), (args.l3, 3), (args.l4, 4), (args.l5, 5)]
            Rs = [[(args.atomic_nums, 0)]] + [Rs] * args.enc_L + [[(args.ncg, 0)]]

        self.layers = torch.nn.ModuleList(
            [GatedBlock(Rs_in, Rs_out, ACTS[args.scalar_act], ACTS[args.gate_act], C)
             for Rs_in, Rs_out in zip(Rs[:-1], Rs[1:-1])] +
            [C(Rs[-2], Rs[-1])]
        )
        self.Rs = Rs

    def forward(self, features, geometry):
        output = features
        for layer in self.layers:
            output = layer(output.div(geometry.size(1) ** 0.5), geometry)  # Normalization of layers by number of atoms.
        return output


class Decoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        radial_model = partial(
            CosineBasisModel,
            max_radius=args.rad_maxr,
            number_of_basis=args.rad_nb,
            h=args.rad_h,
            L=args.rad_L,
            act=ACTS[args.rad_act]
        )
        K = partial(Kernel, RadialModel=radial_model)
        C = partial(Convolution, K)

        Rs = [(args.l0, 0), (args.l1, 1), (args.l2, 2), (args.l3, 3), (args.l4, 4), (args.l5, 5)]
        Rs = [[(args.ncg, 0)]] + [Rs] * args.dec_L
        Rs += [[(mul, l) for l, mul in enumerate([1] * (args.proj_lmax + 1))] * args.atomic_nums]

        self.layers = torch.nn.ModuleList(
            [GatedBlock(Rs_in, Rs_out, ACTS[args.scalar_act], ACTS[args.gate_act], C)
             for Rs_in, Rs_out in zip(Rs[:-1], Rs[1:-1])] +
            [C(Rs[-2], Rs[-1])]
        )
        self.Rs = Rs

    def forward(self, features, geometry):
        output = features
        for layer in self.layers:
            output = layer(output.div(geometry.size(1) ** 0.5), geometry)  # Normalization of layers by number of atoms.
        return output


def execute(args):
    # Data
    atomic_num_to_onehot = {'H': [1, 0], 'C': [0, 1]}
    geometries, forces, atomic_nums = load_data(args)
    geometries = torch.from_numpy(geometries).to(device=DEVICE, dtype=PRECISION)
    forces = torch.from_numpy(forces).to(device=DEVICE, dtype=PRECISION)
    features = torch.tensor(list(map(lambda x: atomic_num_to_onehot[x], atomic_nums)), device=DEVICE, dtype=PRECISION)
    features = features.expand(geometries.size(0), -1, -1)

    cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=PRECISION, device=DEVICE)
    cg_features.scatter_(-1, torch.arange(args.ncg, device=DEVICE).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    # Neural Network
    encoder = Encoder(args).to(device=DEVICE)
    decoder = Decoder(args).to(device=DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Temperature scheduler
    temp_schedule = np.linspace(0, args.epochs, args.epochs)
    decay_epoch = int(args.epochs * args.tdr)
    temp_schedule = args.temp * np.exp(-temp_schedule / decay_epoch) + args.tmin
    temp_schedule = torch.from_numpy(temp_schedule).to(device=DEVICE)

    # Forward
    n_atoms = geometries.size(1)
    n_features = features.size(-1)
    n_batches = int(geometries.shape[0] // args.bs)
    n_samples = n_batches * args.bs
    geometries = geometries[:n_samples].reshape(n_batches, args.bs, n_atoms, 3)
    forces = forces[:n_samples].reshape(n_batches, args.bs, n_atoms, 3)
    features = features[:n_samples].reshape(n_batches, args.bs, n_atoms, n_features)

    dynamics = []
    wall_start = perf_counter()
    for epoch in tqdm(range(args.epochs)):
        for step, batch in tqdm(enumerate(torch.randperm(n_batches, device=DEVICE))):
            wall = perf_counter() - wall_start
            if wall > args.wall:
                break

            feat, geo, force = features[batch], geometries[batch], forces[batch]

            # Auto encoder
            logits = encoder(feat, geo)
            cg_assign, st_cg_assign = gumbel_softmax(logits, temp_schedule[epoch])
            cg_xyz = torch.einsum('zij,zik->zjk', cg_assign, geo)

            # End goal is projection of atoms by atomic number onto coarse grained atom.
            # Project every atom onto each CG. Mask by straight-through cg assignment. Split into channels by atomic number.
            relative_xyz = cg_xyz.unsqueeze(1).cpu().detach() - geo.unsqueeze(2).cpu().detach()
            cg_proj = project_to_ylm(relative_xyz, l_max=args.proj_lmax, dtype=PRECISION,
                                     device=DEVICE)  # B, n_atoms, n_cg, sph
            cg_proj = st_cg_assign.unsqueeze(-1) * cg_proj  # B, n_atoms, n_cg, sph
            cg_proj = cg_proj[..., None, :] * feat[..., None, :, None]  # B, n_atoms, n_cg, atomic_numbers, sph
            cg_proj = cg_proj.sum(1)  # B, n_cg, atomic_numbers, sph

            pred_sph = decoder(cg_features, cg_xyz.clone().detach())
            cg_proj = cg_proj.reshape_as(pred_sph)
            loss_ae = criterion(cg_proj, pred_sph)

            # Force matching
            cg_force_assign, _ = gumbel_softmax(logits, temp_schedule[epoch] * 0.7, device=DEVICE, dtype=PRECISION)
            cg_force = torch.einsum('zij,zik->zjk', cg_force_assign, force)
            loss_fm = cg_force.pow(2).sum(2).mean()

            if epoch >= args.fm_epoch:
                loss = loss_ae + args.fm_co * loss_fm
            else:
                loss = loss_ae

            dynamics.append({
                'loss_ae': loss_ae.item(),
                'loss_fm': loss_fm.item(),
                'epoch': epoch,
                'step': step,
                'temp': temp_schedule[epoch],
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        'args': args,
        'dynamics': dynamics,
        # 'train': {
        #     'pred': evaluate(f, features, geometry, train[:len(test)]),
        #     'true': forces[train[:len(test)]],
        # },
        # 'test': {
        #     'pred': evaluate(f, features, geometry, test[:len(train)]),
        #     'true': forces[test[:len(train)]],
        # },
        'encoder': encoder.state_dict() if args.save_state else None,
        'decoder': decoder.state_dict() if args.save_state else None,
    }


def experiment():
    print("Experimenting...")

    # Data
    num_atoms = 32
    rands = torch.randint(args.atomic_nums, (args.bs, num_atoms, 1), dtype=torch.long, device=DEVICE)
    features = torch.zeros(args.bs, num_atoms, args.atomic_nums, dtype=PRECISION, device=DEVICE)
    features.scatter_(-1, rands, 1.0)
    geometries = torch.randn(args.bs, num_atoms, 3, dtype=PRECISION, device=DEVICE)

    # h2o = torch.tensor([[0.0000, 0.0000, 0.0000],
    #                     [0.7586, 0.0000, 0.5043],
    #                     [0.7586, 0.0000, -0.5043]], dtype=PRECISION, device=DEVICE)
    # feat = torch.tensor([[1, 0],
    #                      [0, 1],
    #                      [0, 1]], dtype=PRECISION, device=DEVICE)
    # geometries = torch.cat([h2o, h2o + torch.ones(3)]).unsqueeze(0)
    # features = torch.cat([feat, feat]).unsqueeze(0)

    cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=PRECISION, device=DEVICE)
    cg_features.scatter_(-1, torch.arange(args.ncg, device=DEVICE).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    # Neural Network
    encoder = Encoder(args).to(DEVICE)
    decoder = Decoder(args).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Forward
    losses = []
    for _ in tqdm(range(100)):
        logits = encoder(features, geometries)
        cg_assign, st_cg_assign = gumbel_softmax(logits, 4.0)
        cg_xyz = torch.einsum('zij,zik->zjk', cg_assign, geometries)

        # End goal is projection of atoms by atomic number onto coarse grained atom.
        # Project every atom onto each CG. Mask by straight-through cg assignment. Split into channels by atomic number.
        relative_xyz = cg_xyz.unsqueeze(1).cpu().detach() - geometries.unsqueeze(2).cpu().detach()
        cg_proj = project_to_ylm(relative_xyz, l_max=args.proj_lmax, dtype=PRECISION, device=DEVICE)  # B, n_atoms, n_cg, sph
        cg_proj = st_cg_assign.unsqueeze(-1) * cg_proj  # B, n_atoms, n_cg, sph
        cg_proj = cg_proj[..., None, :] * features[..., None, :, None]  # B, n_atoms, n_cg, atomic_numbers, sph
        cg_proj = cg_proj.sum(1)  # B, n_cg, atomic_numbers, sph

        pred_sph = decoder(cg_features, cg_xyz.clone().detach())
        cg_proj = cg_proj.reshape_as(pred_sph)

        loss = criterion(cg_proj, pred_sph)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Done experimenting.')


def main():
    results = execute(args)
    with open(args.pickle, 'wb') as f:
        torch.save(results, f)


if __name__ == '__main__':
    if args.experiment:
        experiment()
    main()
