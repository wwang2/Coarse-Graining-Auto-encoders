import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

import argparse

from cgae.cgae import load_data, otp_element_to_onehot


def cgae_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--nte", type=int, default=1000, help="Number of test examples.")
    parser.add_argument("--ntr", type=int, default=3000, help="Number of training examples.")

    # cgae hyper parameters
    parser.add_argument("--atomic_nums", type=int, default=2, help="Count of possible atomic numbers.")
    parser.add_argument("--ncg", type=int, default=3, help="Count of coarse grained united atoms.")
    parser.add_argument("--temp", type=float, default=4.0, help="Set initial temperature.")
    parser.add_argument("--tmin", type=float, default=0.2, help="Set minimum temperature.")
    parser.add_argument("--tdr", type=float, help="Temperature decay rate. Normally set automatically")
    parser.add_argument("--fm_epoch", type=int, default=400, help="Which epoch should force matching being.")
    parser.add_argument("--fm_co", type=float, default=1.0, help="Coefficient multiplied by force matching loss.")

    # General hyper parameters
    parser.add_argument("--bs", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")

    # Calculation
    parser.add_argument("--cpu", action='store_true', help="Force calculation on cpu.")
    parser.add_argument("--gpu", type=int, default=0, choices=list(range(torch.cuda.device_count())), help="Which gpu?")
    parser.add_argument("--double", action='store_true', help="Calculate using double precision.")
    parser.add_argument("--wall", type=float, default=5 * 60, help="If calc time is too long, break.")
    parser.add_argument("--epochs", type=int, default=800, help="Number of epochs to calculate.")
    parser.add_argument("--seed", type=int, default=42, help="Set seed before batching.")

    # Saving
    parser.add_argument("--pickle", type=str, default='out.pkl', help="File for results.")
    parser.add_argument("--save_state", action='store_true', help="Save encoder and decoder state. Default False.")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    args.precision = torch.float64 if args.double else torch.float32
    gpu = torch.cuda.is_available() and not args.cpu
    args.device = torch.device(f"cuda:{args.gpu}") if gpu else torch.device("cpu")
    print(f"Calculating on {args.device}.")
    return args


def data(args):
    geometries, forces, elements = load_data(args.ntr)
    geometries = torch.from_numpy(geometries).to(device=args.device, dtype=args.precision)
    forces = torch.from_numpy(forces).to(device=args.device, dtype=args.precision)
    features = torch.tensor(list(map(lambda x: otp_element_to_onehot()[x], elements)),
                            device=args.device, dtype=args.precision)
    features = features.expand(geometries.size(0), -1, -1)
    return geometries, forces, features


def batch(geometries, forces, features, args):
    n_atoms = geometries.size(1)
    n_features = features.size(-1)
    n_batches = int(geometries.shape[0] // args.bs)
    n_samples = n_batches * args.bs
    geometries = geometries[:n_samples].reshape(n_batches, args.bs, n_atoms, 3)
    forces = forces[:n_samples].reshape(n_batches, args.bs, n_atoms, 3)
    features = features[:n_samples].reshape(n_batches, args.bs, n_atoms, n_features)
    return n_batches, geometries, forces, features
