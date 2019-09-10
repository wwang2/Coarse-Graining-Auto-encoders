import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

import numpy as np
from time import perf_counter
from tqdm import tqdm
from argparse import ArgumentParser

from se3cnn.non_linearities import rescaled_act
from se3cnn.SO3 import spherical_harmonics_xyz

from cgae.gs import gumbel_softmax
from cgae.equi import Encoder, Decoder, ACTS
from cgae.arguments import cgae_parser
from cgae.data import load_data

parser = ArgumentParser(parents=[cgae_parser()])
parser.add_argument("--scalar_encoder", action='store_true', help="Set the encoder to only deal in scalar values.")
parser.add_argument("--scalar_act", type=str, default="relu", choices=ACTS, help="Select scalar activation.")
parser.add_argument("--gate_act", type=str, default="sigmoid", choices=ACTS, help="Select gate activation.")
parser.add_argument("--softplus_beta", type=float, default=1.0, help="Which beta for softplus and shifted softplus?")
parser.add_argument("--l0", type=int, default=5)
parser.add_argument("--enc_l0", type=int, default=5, help="l0 multiplicity for the encoder.")
parser.add_argument("--l1", type=int, default=5)
parser.add_argument("--l2", type=int, default=5)
parser.add_argument("--l3", type=int, default=5)
parser.add_argument("--l4", type=int, default=5)
parser.add_argument("--l5", type=int, default=5)
parser.add_argument("--rad_act", type=str, default="relu", choices=ACTS, help="Select radial activation.")
parser.add_argument("--rad_nb", type=int, default=20, help="Radial number of bases.")
parser.add_argument("--rad_maxr", type=float, default=3, help="Max radius.")
parser.add_argument("--rad_h", type=int, default=50, help="Size of radial weight parameters.")
parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
parser.add_argument("--proj_lmax", type=int, default=5, help="What is the l max for projection.")
parser.add_argument("-e", "--experiment", action='store_true', help="Run experiment function.")
args = parser.parse_args()
args.precision = torch.float64 if args.double else torch.float32
args.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print(f"Calculating on {args.device}.")

ACTS['softplus'] = rescaled_act.Softplus(args.softplus_beta)
ACTS['shifted_softplus'] = rescaled_act.ShiftedSoftplus(args.softplus_beta)


def project_to_ylm(relative_coords, l_max=5, dtype=None, device=None):
    sh = spherical_harmonics_xyz(range(l_max + 1), relative_coords, dtype=dtype, device=device)
    rank = len(sh.shape)
    return sh.permute(*range(1, rank), 0)


# def autoencoder(args):
#     pass
#
#
# def evaluate(f, features, geometry, indicies):
#     with torch.no_grad():
#         outs = []
#         for i in tqdm(range(0, len(indicies), 50), file=sys.stdout):
#             sys.stdout.flush()
#             batch = indicies[i: i + 50]
#             out = f(features[batch], geometry[batch])  # [batch, atom, xyz]
#             outs.append(out)
#         return torch.cat(outs)


def execute(args):
    # Data
    atomic_num_to_onehot = {'H': [1, 0], 'C': [0, 1]}
    geometries, forces, atomic_nums = load_data(args)
    geometries = torch.from_numpy(geometries).to(device=args.device, dtype=args.precision)
    forces = torch.from_numpy(forces).to(device=args.device, dtype=args.precision)
    features = torch.tensor(list(map(lambda x: atomic_num_to_onehot[x], atomic_nums)),
                            device=args.device, dtype=args.precision)
    features = features.expand(geometries.size(0), -1, -1)

    cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=args.precision, device=args.device)
    cg_features.scatter_(-1, torch.arange(args.ncg, device=args.device).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    # Neural Network
    encoder = Encoder(args).to(device=args.device)
    decoder = Decoder(args).to(device=args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Temperature scheduler
    temp_schedule = np.linspace(0, args.epochs, args.epochs)
    decay_epoch = int(args.epochs * args.tdr)
    temp_schedule = args.temp * np.exp(-temp_schedule / decay_epoch) + args.tmin
    temp_schedule = torch.from_numpy(temp_schedule).to(device=args.device)

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
        for step, batch in tqdm(enumerate(torch.randperm(n_batches, device=args.device))):
            wall = perf_counter() - wall_start
            if wall > args.wall:
                break

            feat, geo, force = features[batch], geometries[batch], forces[batch]

            # Auto encoder
            logits = encoder(feat, geo)
            cg_assign, st_cg_assign = gumbel_softmax(logits, temp_schedule[epoch])
            cg_xyz = torch.einsum('zij,zik->zjk', cg_assign, geo)

            # End goal is projection of atoms by atomic number onto coarse grained atom.
            # Project every atom onto each CG.
            # Mask by straight-through cg assignment.
            # Split into channels by atomic number.
            relative_xyz = cg_xyz.unsqueeze(1).cpu().detach() - geo.unsqueeze(2).cpu().detach()
            cg_proj = project_to_ylm(relative_xyz, l_max=args.proj_lmax, dtype=args.precision,
                                     device=args.device)  # B, n_atoms, n_cg, sph
            cg_proj = st_cg_assign.unsqueeze(-1) * cg_proj  # B, n_atoms, n_cg, sph
            cg_proj = cg_proj[..., None, :] * feat[..., None, :, None]  # B, n_atoms, n_cg, atomic_numbers, sph
            cg_proj = cg_proj.sum(1)  # B, n_cg, atomic_numbers, sph

            pred_sph = decoder(cg_features, cg_xyz.clone().detach())
            cg_proj = cg_proj.reshape_as(pred_sph)
            loss_ae = criterion(cg_proj, pred_sph)

            # Force matching
            cg_force_assign, _ = gumbel_softmax(logits, temp_schedule[epoch] * 0.7,
                                                device=args.device, dtype=args.precision)
            cg_force = torch.einsum('zij,zik->zjk', cg_force_assign, force)
            loss_fm = cg_force.pow(2).sum(2).mean()

            if epoch >= args.fm_epoch:
                loss = loss_ae + args.fm_co * loss_fm
            else:
                loss = loss_ae

            dynamics.append({
                'loss_ae': loss_ae.item(),
                'loss_fm': loss_fm.item(),
                'loss': loss,
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
    rands = torch.randint(args.atomic_nums, (args.bs, num_atoms, 1), dtype=torch.long, device=args.device)
    features = torch.zeros(args.bs, num_atoms, args.atomic_nums, dtype=args.precision, device=args.device)
    features.scatter_(-1, rands, 1.0)
    geometries = torch.randn(args.bs, num_atoms, 3, dtype=args.precision, device=args.device)

    # h2o = torch.tensor([[0.0000, 0.0000, 0.0000],
    #                     [0.7586, 0.0000, 0.5043],
    #                     [0.7586, 0.0000, -0.5043]], dtype=args.precision, device=args.device)
    # feat = torch.tensor([[1, 0],
    #                      [0, 1],
    #                      [0, 1]], dtype=args.precision, device=args.device)
    # geometries = torch.cat([h2o, h2o + torch.ones(3)]).unsqueeze(0)
    # features = torch.cat([feat, feat]).unsqueeze(0)

    cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=args.precision, device=args.device)
    cg_features.scatter_(-1, torch.arange(args.ncg, device=args.device).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    # Neural Network
    encoder = Encoder(args).to(args.device)
    decoder = Decoder(args).to(args.device)
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
        cg_proj = project_to_ylm(relative_xyz, l_max=args.proj_lmax, dtype=args.precision, device=args.device)  # B, n_atoms, n_cg, sph
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
