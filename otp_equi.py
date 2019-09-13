import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from time import perf_counter
from tqdm import tqdm
from argparse import ArgumentParser

from se3cnn.non_linearities import rescaled_act
from se3cnn.SO3 import spherical_harmonics_xyz

from cgae.gs import gumbel_softmax
from cgae.cgae import temp_scheduler
from cgae.equi import Encoder, Decoder, ACTS

import otp


def add_args(parent_parser, add_help):
    parser = ArgumentParser(parents=[parent_parser], add_help=add_help)
    parser.add_argument("--scalar_encoder", action='store_true', help="Set the encoder to only deal in scalar values.")
    parser.add_argument("--scalar_act", type=str, default="relu", choices=ACTS, help="Select scalar activation.")
    parser.add_argument("--gate_act", type=str, default="sigmoid", choices=ACTS, help="Select gate activation.")
    parser.add_argument("--softplus_beta", type=float, default=1.0,
                        help="Which beta for softplus and shifted softplus?")
    parser.add_argument("--l0", type=int, default=5)
    parser.add_argument("--enc_l0", type=int, default=5, help="l0 multiplicity for the encoder.")
    parser.add_argument("--l1", type=int, default=5)
    parser.add_argument("--l2", type=int, default=5)
    parser.add_argument("--l3", type=int, default=5)
    parser.add_argument("--l4", type=int, default=5)
    parser.add_argument("--l5", type=int, default=5)
    parser.add_argument("--enc_L", type=int, default=1, help="How many layers to create for the encoder.")
    parser.add_argument("--dec_L", type=int, default=1, help="How many layers to create for the decoder.")
    parser.add_argument("--rad_act", type=str, default="relu", choices=ACTS, help="Select radial activation.")
    parser.add_argument("--rad_nb", type=int, default=20, help="Radial number of bases.")
    parser.add_argument("--rad_maxr", type=float, default=3, help="Max radius.")
    parser.add_argument("--rad_h", type=int, default=50, help="Size of radial weight parameters.")
    parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
    parser.add_argument("--proj_lmax", type=int, default=5, help="What is the l max for projection.")
    parser.add_argument("--gumble_sm_proj", action='store_true',
                        help="For the target projection, use a gumble softmax sample instead of a straight-through "
                             "gumble softmax sample.")
    return parser


parser = add_args(otp.cgae_parser(), add_help=True)
args = otp.parse_args(parser)

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


def project_onto_cg(r, assignment, feature_mask, args):
    # Project every atom onto each CG.
    # Mask by straight-through cg assignment.
    # Split into channels by atomic number.
    cg_proj = project_to_ylm(r, l_max=args.proj_lmax, dtype=args.precision, device=args.device)  # B, n_atoms, n_cg, sph
    cg_proj = assignment.unsqueeze(-1) * cg_proj  # B, n_atoms, n_cg, sph
    cg_proj = cg_proj[..., None, :] * feature_mask[..., None, :, None]  # B, n_atoms, n_cg, atomic_numbers, sph
    cg_proj = cg_proj.sum(1)  # B, n_cg, atomic_numbers, sph
    return cg_proj


def execute(args):
    geometries, forces, features = otp.data(args)

    cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=args.precision, device=args.device)
    cg_features.scatter_(-1, torch.arange(args.ncg, device=args.device).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    encoder = Encoder(args).to(device=args.device)
    decoder = Decoder(args).to(device=args.device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    temp_sched = temp_scheduler(args.epochs, args.tdr, args.temp, args.tmin, dtype=args.precision, device=args.device)
    n_batches, geometries, forces, features = otp.batch(geometries, forces, features, args)

    dynamics = []
    summaries = []
    wall_start = perf_counter()
    torch.manual_seed(args.seed)
    for epoch in tqdm(range(args.epochs)):
        for step, batch in tqdm(enumerate(torch.randperm(n_batches, device=args.device))):
            feat, geo, force = features[batch], geometries[batch], forces[batch]

            # Auto encoder
            logits = encoder(feat, geo)
            cg_assign, st_cg_assign = gumbel_softmax(logits, temp_sched[epoch],
                                                     dtype=args.precision, device=args.device)
            E = cg_assign / cg_assign.sum(1).unsqueeze(1)
            cg_xyz = torch.einsum('zij,zik->zjk', E, geo)

            # End goal is projection of atoms by atomic number onto coarse grained atom.
            relative_xyz = cg_xyz.unsqueeze(1).cpu().detach() - geo.unsqueeze(2).cpu().detach()
            if args.gumble_sm_proj:
                cg_proj = project_onto_cg(relative_xyz, cg_assign, feat, args)
            else:
                cg_proj = project_onto_cg(relative_xyz, st_cg_assign, feat, args)

            pred_sph = decoder(cg_features, cg_xyz.clone().detach())
            cg_proj = cg_proj.reshape_as(pred_sph)
            loss_ae = (cg_proj - pred_sph).pow(2).sum(-1).div(args.atomic_nums).mean()

            # Force matching
            cg_force_assign, _ = gumbel_softmax(logits, temp_sched[epoch] * 0.7,
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
                'loss': loss.item(),
                'epoch': epoch,
                'step': step,
                'batch': batch.item(),
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wall = perf_counter() - wall_start
        if wall > args.wall:
            break
            
        summaries.append({
            'loss_ae': loss_ae.item(),
            'loss_fm': loss_fm.item(),
            'loss': loss.item(),
            'epoch': epoch,
            'step': step,
            'batch': batch.item(),
            'cg_xyz': cg_xyz,
            'pred_sph': pred_sph,
            'sph': cg_proj,
            'temp': temp_sched[epoch].item(),
            'gumble': cg_assign,
            'st_gumble': st_cg_assign,
        })

    return {
        'args': args,
        'dynamics': dynamics,
        'summaries': summaries,
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


def main():
    results = execute(args)
    with open(args.pickle, 'wb') as f:
        torch.save(results, f)


if __name__ == '__main__':
    main()
