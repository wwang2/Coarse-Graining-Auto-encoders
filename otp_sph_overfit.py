import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from time import perf_counter
from tqdm import tqdm
from argparse import ArgumentParser

from se3cnn.SO3 import spherical_harmonics_xyz

from cgae.gs import gumbel_softmax
from cgae.cgae import temp_scheduler
import cgae.cgae_dense as dense
import cgae.equi as equi

import otp

parser = ArgumentParser(parents=[otp.otp_parser(), otp.otp_equi_parser()])
parser.add_argument(
    "--dense",
    type=str,
    required=True,
    help="Pickle dict with 'encoder' and 'decoder' keys.",
)
args = otp.parse_args(parser)


def single_example(args):
    cg_xyz = torch.tensor(
        [
            [
                [2.1636813, 0.41315195, 2.7376754],
                [-2.2999983, 0.75210804, 4.8020134],
                [0.03099308, 4.3005924, 3.2575095],
            ]
        ],
        device=args.device,
        dtype=args.precision,
    )

    geo = torch.tensor(
        [
            [
                [2.95687, -0.155487, 2.11914],
                [2.06787, 0.671653, 1.43503],
                [0.850204, 1.00373, 2.03273],
                [0.601561, 0.628464, 3.40416],
                [1.56987, -0.160152, 4.10868],
                [2.7057, -0.572914, 3.4475],
                [-0.660677, 0.987154, 4.01818],
                [-1.5062, -0.0960717, 4.51092],
                [-2.79988, 0.102162, 5.03003],
                [-3.24092, 1.39391, 5.26613],
                [-2.4742, 2.51713, 4.79282],
                [-1.26258, 2.37251, 4.09869],
                [-0.543147, 3.54105, 3.57502],
                [0.705632, 3.706, 4.07958],
                [1.4324, 4.92182, 3.79255],
                [0.943385, 5.85763, 2.93262],
                [-0.394225, 5.73001, 2.50639],
                [-1.12944, 4.62928, 2.87664],
                [3.87787, -0.345135, 1.63161],
                [2.28279, 0.994601, 0.40769],
                [0.0681252, 1.5429, 1.54221],
                [1.31643, -0.405288, 5.13049],
                [3.50865, -0.971784, 4.06418],
                [-1.18795, -1.17369, 4.3918],
                [-3.36382, -0.788193, 5.36065],
                [-4.20054, 1.44102, 5.74857],
                [-2.98015, 3.51865, 4.87225],
                [1.22583, 3.04857, 4.6967],
                [2.46003, 5.08062, 4.05937],
                [1.53019, 6.73809, 2.68288],
                [-0.856573, 6.50135, 1.8472],
                [-2.14819, 4.48423, 2.48099],
            ]
        ],
        device=args.device,
        dtype=args.precision,
    )

    feat = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        device=args.device,
        dtype=args.precision,
    )

    # # The purpose is to select the nearest atom to a cg_atom, project it, give a single atom that feature.
    # relative_xyz = geo.unsqueeze(2) - cg_xyz.unsqueeze(1)
    # nearest_atom_ind = relative_xyz.norm(dim=-1).argmin(1).squeeze()
    # gather_inds = nearest_atom_ind.reshape(3, 1).repeat(1, 3).reshape(1, 1, 3, 3)
    # nearest_atoms = relative_xyz.gather(1, gather_inds)
    # atom_mask = torch.zeros_like(nearest_atoms).scatter(
    #     2, torch.zeros_like(nearest_atoms, dtype=torch.long)[:, :, 0:1, :], 1.0
    # )
    # cg_features = (nearest_atoms * atom_mask).reshape(*cg_xyz.shape[:2], -1)
    # cg_features = spherical_harmonics_xyz(2, cg_features).permute(1, 2, 0)
    # l1_features = torch.ones(
    #     *cg_features.shape[:2], 1, device=args.device, dtype=args.precision
    # )
    # cg_features = torch.cat([l1_features, cg_features], dim=-1)

    # One hot cg
    # cg_features = torch.zeros(args.bs, args.ncg, args.ncg, dtype=args.precision, device=args.device)
    # cg_features.scatter_(-1, torch.arange(args.ncg, device=args.device).expand(args.bs, args.ncg).unsqueeze(-1), 1.0)

    # Rs_in = [[(1, 0), (1, 2)]]
    Rs_in = [
        [
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
        ]
    ]
    decoder = equi.Decoder(args, Rs_in=Rs_in).to(device=args.device)
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=args.lr)

    dynamics = []
    summaries = []
    wall_start = perf_counter()
    torch.manual_seed(args.seed)
    for step in tqdm(range(1000)):
        # End goal is projection of atoms by atomic number onto coarse grained atom.
        relative_xyz = (
            geo.unsqueeze(2).cpu().detach() - cg_xyz.unsqueeze(1).cpu().detach()
        )
        nearest_assign = equi.nearest_assignment(cg_xyz, geo)
        cg_proj = otp.project_onto_cg(relative_xyz, nearest_assign, feat, args)

        cg_features = cg_proj.reshape(*cg_proj.shape[:-2], -1).clone()
        pred_sph = decoder(cg_features, cg_xyz.clone().detach())
        cg_proj = cg_proj.reshape_as(pred_sph)
        loss_ae_equi = (cg_proj - pred_sph).pow(2).sum(-1).div(args.atomic_nums).mean()

        loss = loss_ae_equi

        dynamics.append(
            {"loss_ae_equi": loss_ae_equi.item(), "loss": loss.item(), "step": step}
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wall = perf_counter() - wall_start
        if wall > args.wall:
            break

        summaries.append(
            {
                "loss_ae_equi": loss_ae_equi.item(),
                "loss": loss.item(),
                "step": step,
                "cg_xyz": cg_xyz,
                "pred_sph": pred_sph,
                "sph": cg_proj,
                "nearest": nearest_assign,
            }
        )

    return {
        "args": args,
        "dynamics": dynamics,
        "summaries": summaries,
        # 'train': {
        #     'pred': evaluate(f, features, geometry, train[:len(test)]),
        #     'true': forces[train[:len(test)]],
        # },
        # 'test': {
        #     'pred': evaluate(f, features, geometry, test[:len(train)]),
        #     'true': forces[test[:len(train)]],
        # },
        # 'encoder': encoder.state_dict() if args.save_state else None,
        "decoder": decoder.state_dict() if args.save_state else None,
    }


def execute(args):
    dense_dict = torch.load(args.dense, map_location=args.device)
    geometries, forces, features = otp.data(args)

    encoder_dense = dense.Encoder(
        in_dim=geometries.size(1), out_dim=args.ncg, device=args.device
    ).to(args.device)
    encoder_dense.load_state_dict(dense_dict["encoder"])
    encoder_dense.weight1.detach_()
    decoder_dense = dense.Decoder(in_dim=args.ncg, out_dim=geometries.size(1)).to(
        args.device
    )
    decoder_dense.load_state_dict(dense_dict["decoder"])
    decoder_dense.weight.detach_()

    if args.cg_ones:
        cg_features = torch.ones(
            args.bs, args.ncg, 1, dtype=args.precision, device=args.device
        )
    elif args.cg_specific_atom:
        pass
    else:
        cg_features = torch.zeros(
            args.bs, args.ncg, args.ncg, dtype=args.precision, device=args.device
        )
        cg_features.scatter_(
            -1,
            torch.arange(args.ncg, device=args.device)
            .expand(args.bs, args.ncg)
            .unsqueeze(-1),
            1.0,
        )

    # Encoder... TBD
    decoder = equi.Decoder(args).to(device=args.device)
    # optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=args.lr)
    temp_sched = temp_scheduler(
        args.epochs,
        args.tdr,
        args.temp,
        args.tmin,
        dtype=args.precision,
        device=args.device,
    )
    n_batches, geometries, forces, features = otp.batch(
        geometries, forces, features, args.bs
    )

    dynamics = []
    summaries = []
    wall_start = perf_counter()
    torch.manual_seed(args.seed)
    for epoch in tqdm(range(args.epochs)):
        # for step, batch in tqdm(enumerate(torch.randperm(n_batches, device=args.device))):
        for step, batch in tqdm(
            enumerate([torch.tensor(0, device=args.device)] * n_batches)
        ):
            feat, geo, force = features[batch], geometries[batch], forces[batch]

            # Auto encoder
            cg_xyz = encoder_dense(geo, temp_sched[epoch])
            logits = encoder_dense.weight1.t()
            cg_assign, st_cg_assign = gumbel_softmax(
                logits, temp_sched[epoch], dtype=args.precision, device=args.device
            )
            decoded = decoder_dense(cg_xyz)
            loss_ae_dense = (decoded - geo).pow(2).sum(-1).mean()

            # End goal is projection of atoms by atomic number onto coarse grained atom.
            relative_xyz = (
                geo.unsqueeze(2).cpu().detach() - cg_xyz.unsqueeze(1).cpu().detach()
            )
            nearest_assign = equi.nearest_assignment(cg_xyz, geo)
            if args.gumble_sm_proj:
                cg_proj = otp.project_onto_cg(relative_xyz, cg_assign, feat, args)
            elif args.nearest:
                cg_proj = otp.project_onto_cg(relative_xyz, nearest_assign, feat, args)
            else:
                cg_proj = otp.project_onto_cg(relative_xyz, st_cg_assign, feat, args)

            pred_sph = decoder(cg_features, cg_xyz.clone().detach())
            cg_proj = cg_proj.reshape_as(pred_sph)
            loss_ae_equi = (
                (cg_proj - pred_sph).pow(2).sum(-1).div(args.atomic_nums).mean()
            )

            if args.fm and epoch >= args.fm_epoch:
                # Force matching
                cg_force_assign, _ = gumbel_softmax(
                    logits,
                    temp_sched[epoch] * args.force_temp_coeff,
                    device=args.device,
                    dtype=args.precision,
                )
                cg_force = torch.einsum("...ij,zik->zjk", cg_force_assign, force)
                loss_fm = cg_force.pow(2).sum(-1).mean()

                # loss = loss_ae_equi + loss_ae_dense + args.fm_co * loss_fm
            else:
                loss_fm = torch.tensor(0)
                # loss = loss_ae_equi + loss_ae_dense

            loss = loss_ae_equi

            dynamics.append(
                {
                    "loss_ae_equi": loss_ae_equi.item(),
                    "loss_ae_dense": loss_ae_dense.item(),
                    "loss_fm": loss_fm.item(),
                    "loss": loss.item(),
                    "epoch": epoch,
                    "step": step,
                    "batch": batch.item(),
                }
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wall = perf_counter() - wall_start
        if wall > args.wall:
            break

        summaries.append(
            {
                "loss_ae_equi": loss_ae_equi.item(),
                "loss_ae_dense": loss_ae_dense.item(),
                "loss_fm": loss_fm.item(),
                "loss": loss.item(),
                "epoch": epoch,
                "step": step,
                "batch": batch.item(),
                "cg_xyz": cg_xyz,
                "pred_sph": pred_sph,
                "sph": cg_proj,
                "temp": temp_sched[epoch].item(),
                "gumble": cg_assign,
                "st_gumble": st_cg_assign,
                "nearest": nearest_assign,
            }
        )

    return {
        "args": args,
        "dynamics": dynamics,
        "summaries": summaries,
        # 'train': {
        #     'pred': evaluate(f, features, geometry, train[:len(test)]),
        #     'true': forces[train[:len(test)]],
        # },
        # 'test': {
        #     'pred': evaluate(f, features, geometry, test[:len(train)]),
        #     'true': forces[test[:len(train)]],
        # },
        # 'encoder': encoder.state_dict() if args.save_state else None,
        "decoder": decoder.state_dict() if args.save_state else None,
    }


def main():
    # results = execute(args)
    # with open(args.pickle, 'wb') as f:
    #     torch.save(results, f)
    ok = single_example(args)
    with open(args.pickle, "wb") as f:
        torch.save(ok, f)


if __name__ == "__main__":
    main()
