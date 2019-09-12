import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from time import perf_counter
from tqdm import tqdm
from argparse import ArgumentParser

# from cgae.utils import write_traj, save_traj
from cgae.cgae_dense import gumbel_softmax, Encoder, Decoder
from cgae.cgae import temp_scheduler

import otp

parser = ArgumentParser(parents=[otp.cgae_parser()])
args = otp.parse_args(parser)


def execute(args):
    # Data
    geometries, forces, features = otp.data(args)

    encoder = Encoder(in_dim=geometries.size(1), out_dim=args.ncg, hard=False, device=args.device).to(args.device)
    decoder = Decoder(in_dim=args.ncg, out_dim=geometries.size(1)).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    temp_sched = temp_scheduler(args.epochs, args.tdr, args.temp, args.tmin, dtype=args.precision, device=args.device)
    n_batches, geometries, forces, features = otp.batch(geometries, forces, features, args)

    dynamics = []
    wall_start = perf_counter()
    torch.manual_seed(args.seed)
    for epoch in tqdm(range(args.epochs)):
        for step, batch in tqdm(enumerate(torch.randperm(n_batches, device=args.device))):
            wall = perf_counter() - wall_start
            if wall > args.wall:
                break

            _, geo, force = features[batch], geometries[batch], forces[batch]
            cg_xyz = encoder(geo, temp_sched[epoch])
            CG = gumbel_softmax(encoder.weight1.t(), temp_sched[epoch] * 0.7, device=args.device).t()
            decoded = decoder(cg_xyz)
            loss_ae = criterion(decoded, geo)

            f = torch.matmul(CG, force)
            loss_fm = f.pow(2).sum(2).mean()

            if epoch >= args.fm_epoch:
                loss = loss_ae + args.fm_co * loss_fm
            else:
                loss = loss_ae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dynamics.append({
                'loss_ae': loss_ae.item(),
                'loss_fm': loss_fm.item(),
                'loss': loss,
                'epoch': epoch,
                'step': step,
                'temp': temp_sched[epoch],
                'gumble': gumbel_softmax(encoder.weight1.t(), temp_sched[epoch], device='cpu'),
                'cg_xyz': cg_xyz
            })

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


def main():
    results = execute(args)
    with open(args.pickle, 'wb') as f:
        torch.save(results, f)


if __name__ == '__main__':
    main()
