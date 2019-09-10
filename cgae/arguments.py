import torch
import argparse


def cgae_parser():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--nte", type=int, default=1000, help="Number of test examples.")
    parser.add_argument("--ntr", type=int, default=3000, help="Number of training examples.")

    # cgae hyper parameters
    parser.add_argument("--atomic_nums", type=int, default=2, help="Count of possible atomic numbers.")
    parser.add_argument("--ncg", type=int, default=3, help="Count of coarse grained united atoms.")
    parser.add_argument("--temp", type=float, default=4.0, help="Set initial temperature.")
    parser.add_argument("--tmin", type=float, default=0.2, help="Set minimum temperature.")
    parser.add_argument("--tdr", type=float, default=0.4, help="Temperature decay rate.")
    parser.add_argument("--fm_epoch", type=int, default=400, help="Which epoch should force matching being.")
    parser.add_argument("--fm_co", type=float, default=0.005, help="Coefficient multiplied by force matching loss.")

    # General hyper parameters
    parser.add_argument("--bs", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--enc_L", type=int, default=3, help="How many layers to create for the encoder.")
    parser.add_argument("--dec_L", type=int, default=1, help="How many layers to create for the decoder.")

    # Calculation
    parser.add_argument("--cpu", action='store_true', help="Force calculation on cpu.")
    parser.add_argument("--gpu", type=int, default=0, choices=list(range(torch.cuda.device_count())), help="Which gpu?")
    parser.add_argument("--double", action='store_true', help="Calculate using double precision.")
    parser.add_argument("--wall", type=float, default=5 * 60, help="If calc time is too long, break.")
    parser.add_argument("--epochs", type=int, default=800, help="Number of epochs to calculate.")

    # Saving
    parser.add_argument("--pickle", type=str, default='out.pkl', help="File for results.")
    parser.add_argument("--save_state", action='store_true', help="Save encoder and decoder state. Default False.")
    return parser


def get_precision_device(args):
    if args.double:
        precision = torch.float64
    else:
        precision = torch.float32
    torch.set_default_dtype(precision)

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    return precision, device


def main():
    pass


if __name__ == '__main__':
    main()
