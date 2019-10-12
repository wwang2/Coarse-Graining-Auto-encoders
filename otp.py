import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

import argparse

import se3cnn.SO3 as SO3
from se3cnn.non_linearities import rescaled_act

from cgae.cgae import load_data, otp_element_to_onehot
from cgae.equi import ACTS


def otp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument(
        "--nte", type=int, default=1000, help="Number of test examples."
    )
    parser.add_argument(
        "--ntr", type=int, default=3000, help="Number of training examples."
    )

    # cgae hyper parameters
    parser.add_argument(
        "--atomic_nums", type=int, default=2, help="Count of possible atomic numbers."
    )
    parser.add_argument(
        "--ncg", type=int, default=3, help="Count of coarse grained united atoms."
    )
    parser.add_argument(
        "--temp", type=float, default=4.0, help="Set initial temperature."
    )
    parser.add_argument(
        "--tmin", type=float, default=0.1, help="Set minimum temperature."
    )
    parser.add_argument(
        "--tdr", type=float, help="Temperature decay rate. Normally set automatically"
    )
    parser.add_argument(
        "--fm", action="store_true", help="Turn on force matching at a certain epoch."
    )
    parser.add_argument(
        "--fm_epoch",
        type=int,
        default=400,
        help="Which epoch should force matching being.",
    )
    parser.add_argument(
        "--fm_co",
        type=float,
        default=1.0,
        help="Coefficient multiplied by force matching loss.",
    )

    # General hyper parameters
    parser.add_argument("--bs", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--force_temp_coeff",
        type=float,
        default=0.7,
        help="Gumble is sampled at a diff temp for "
        "forces. This is that coefficient.",
    )

    # Calculation
    parser.add_argument("--cpu", action="store_true", help="Force calculation on cpu.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        choices=list(range(torch.cuda.device_count())),
        help="Which gpu?",
    )
    parser.add_argument(
        "--double", action="store_true", help="Calculate using double precision."
    )
    parser.add_argument(
        "--wall", type=float, default=5 * 60, help="If calc time is too long, break."
    )
    parser.add_argument(
        "--epochs", type=int, default=800, help="Number of epochs to calculate."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Set seed before batching."
    )

    # Saving
    parser.add_argument(
        "--pickle", type=str, default="out.pkl", help="File for results."
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="Save encoder and decoder state. Default False.",
    )
    return parser


def otp_equi_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--high_l_encoder",
        action="store_true",
        help="Set the encoder to include high order sph.",
    )
    parser.add_argument(
        "--scalar_act",
        type=str,
        default="softplus",
        choices=ACTS,
        help="Select scalar activation.",
    )
    parser.add_argument(
        "--gate_act",
        type=str,
        default="sigmoid",
        choices=ACTS,
        help="Select gate activation.",
    )
    parser.add_argument(
        "--softplus_beta",
        type=float,
        default=1.0,
        help="Which beta for softplus and shifted softplus?",
    )
    parser.add_argument("--l0", type=int, default=5)
    parser.add_argument(
        "--enc_l0", type=int, default=20, help="l0 multiplicity for the encoder."
    )
    parser.add_argument("--l1", type=int, default=5)
    parser.add_argument("--l2", type=int, default=5)
    parser.add_argument("--l3", type=int, default=5)
    parser.add_argument("--l4", type=int, default=5)
    parser.add_argument("--l5", type=int, default=5)
    parser.add_argument(
        "--enc_L",
        type=int,
        default=3,
        help="How many layers to create for the encoder.",
    )
    parser.add_argument(
        "--dec_L",
        type=int,
        default=1,
        help="How many layers to create for the decoder.",
    )
    parser.add_argument(
        "--rad_act",
        type=str,
        default="softplus",
        choices=ACTS,
        help="Select radial activation.",
    )
    parser.add_argument(
        "--rad_nb", type=int, default=20, help="Radial number of bases."
    )
    parser.add_argument("--rad_maxr", type=float, default=3, help="Max radius.")
    parser.add_argument(
        "--rad_h", type=int, default=50, help="Size of radial weight parameters."
    )
    parser.add_argument("--rad_L", type=int, default=2, help="Number of radial layers.")
    parser.add_argument(
        "--proj_lmax", type=int, default=5, help="What is the l max for projection."
    )
    projection_group = parser.add_mutually_exclusive_group()
    projection_group.add_argument(
        "--gumble_sm_proj",
        action="store_true",
        help="For the target projection, use a gumble softmax sample instead of a "
        "straight-through gumble softmax sample.",
    )
    projection_group.add_argument(
        "--nearest",
        action="store_true",
        help="Use the nearest atoms for sph projection target.",
    )
    cg_features_group = parser.add_mutually_exclusive_group()
    cg_features_group.add_argument(
        "--cg_ones",
        action="store_true",
        help="cg_features get a single scalar channel. (This didn't help overfit.)",
    )
    cg_features_group.add_argument(
        "--cg_specific_atom",
        type=int,
        choices={1, 2, 3, 4, 5},
        help="cg_features gets the location of an atom. l=?? projection",
    )

    return parser


def parse_args(parser):
    args = parser.parse_args()
    args.precision = torch.float64 if args.double else torch.float32
    gpu = torch.cuda.is_available() and not args.cpu
    args.device = torch.device("cuda:{}".format(args.gpu)) if gpu else torch.device("cpu")

    ACTS["softplus"] = rescaled_act.Softplus(args.softplus_beta)
    ACTS["shifted_softplus"] = rescaled_act.ShiftedSoftplus(args.softplus_beta)

    print("Calculating on {}.".format(args.gpu))
    return args


def project_onto_cg(r, assignment, feature_mask, args, adjusted=False):
    # Project every atom onto each CG.
    # Mask by straight-through cg assignment.
    # Split into channels by atomic number.
    cg_proj = project_to_ylm(
        r,
        l_max=args.proj_lmax,
        dtype=args.precision,
        device=args.device,
        adjusted=adjusted,
    )  # B, n_atoms, n_cg, sph
    cg_proj = assignment.unsqueeze(-1) * cg_proj  # B, n_atoms, n_cg, sph
    cg_proj = (
        cg_proj[..., None, :] * feature_mask[..., None, :, None]
    )  # B, n_atoms, n_cg, atomic_numbers, sph
    cg_proj = cg_proj.sum(1)  # B, n_cg, atomic_numbers, sph
    return cg_proj


def project_to_ylm(relative_coords, l_max=5, dtype=None, device=None, adjusted=False):
    if adjusted:
        by_batch = []
        for b in range(relative_coords.shape[0]):
            proj_on_cg = torch.stack(
                [
                    adjusted_projection(relative_coords[b, :, i, :], l_max)
                    for i in range(relative_coords.shape[2])
                ],
                dim=1,
            )
            by_batch.append(proj_on_cg)
        sh = torch.stack(by_batch)
    else:
        sh = SO3.spherical_harmonics_xyz(
            range(l_max + 1), relative_coords, sph_last=True, dtype=dtype, device=device
        )
    return sh


def adjusted_projection(vectors, L_max, sum=False):
    radii = vectors.norm(2, -1)
    vectors = vectors[radii > 0.0]
    radii = radii[radii > 0.0]

    angles = SO3.xyz_to_angles(vectors)
    coeff = SO3.spherical_harmonics_dirac(L_max, *angles)
    coeff *= radii.unsqueeze(-2)
    # Handle case where only have a center
    if coeff.shape[1] == 0:
        return torch.zeros(coeff.shape[0])

    A = torch.einsum(
        "ia,ib->ab", SO3.spherical_harmonics(range(L_max + 1), *angles), coeff
    )
    coeff *= torch.gels(radii, A).solution.view(-1)
    if sum:
        return coeff.sum(-1)
    else:
        rank = len(coeff.shape)
        return coeff.permute(*range(1, rank), 0)


def project_atom_onto_cg_features(
    relative_coords, order, atom_inds, cg_ind, dtype=None, device=None
):
    sph = SO3.spherical_harmonics_xyz(
        order, relative_coords, sph_last=True, dtype=dtype, device=device
    )
    try:
        count_projected_atoms = len(atom_inds)
    except TypeError:
        count_projected_atoms = 1
    index = (
        torch.tensor(cg_ind, device=device)
        .view(1, 1, 1, 1)
        .expand(sph.size(0), count_projected_atoms, -1, sph.size(-1))
    )
    if not torch.is_tensor(atom_inds):
        atom_inds = torch.tensor(atom_inds, device=device)
    selected_sph = sph.clone().detach().index_select(1, atom_inds)
    mask = torch.zeros_like(selected_sph).scatter_(-2, index, 1.0)
    return (mask * selected_sph).sum(1)


def data(args):
    geometries, forces, elements = load_data(args.ntr)
    geometries = torch.from_numpy(geometries).to(
        device=args.device, dtype=args.precision
    )
    forces = torch.from_numpy(forces).to(device=args.device, dtype=args.precision)
    features = torch.tensor(
        list(map(lambda x: otp_element_to_onehot()[x], elements)),
        device=args.device,
        dtype=args.precision,
    )
    features = features.expand(geometries.size(0), -1, -1)
    return geometries, forces, features


def batch(geometries, forces, features, batch_size):
    n_atoms = geometries.size(1)
    n_features = features.size(-1)
    n_batches = int(geometries.shape[0] // batch_size)
    n_samples = n_batches * batch_size
    geometries = geometries[:n_samples].reshape(n_batches, batch_size, n_atoms, 3)
    forces = forces[:n_samples].reshape(n_batches, batch_size, n_atoms, 3)
    features = features[:n_samples].reshape(n_batches, batch_size, n_atoms, n_features)
    return n_batches, geometries, forces, features


def assign_locally(cg_geo, atoms_geo):
    return (atoms_geo.unsqueeze(-2) - cg_geo.unsqueeze(-3)).pow(2).sum(-1).argmin(1)
