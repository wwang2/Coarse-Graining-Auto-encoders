import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data

from argparse import ArgumentParser
from tqdm import tqdm

from cgae.utils import write_traj, save_traj
from cgae.cgae_dense import gumbel_softmax, Encoder, Decoder
from cgae.cgae import temp_scheduler

import otp

parser = ArgumentParser(parents=[otp.cgae_parser()])
args = otp.parse_args(parser)


def execute(args):
    # Data
    geometries, forces, features = otp.data(args)
    encoder, decoder, criterion, optimizer = otp.neural_network(Encoder, Decoder, args)
    temp_sched = temp_scheduler(args.epochs, args.tdr, args.temp, args.tmin, dtype=args.precision, device=args.device)
    n_batches, geometries, forces, features = otp.batch(geometries, forces, features, args)

    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(xyz):
            batch = torch.Tensor(batch.reshape(-1, n_atom, 3)).to(device)
            cg_xyz = encoder(batch, t_sched[epoch])
            CG = gumbel_softmax(encoder.weight1.t(), t_sched[epoch] * 0.7, device=device).t()

            decoded = decoder(cg_xyz)
            loss_ae = criterion(decoded, batch)

            f0 = torch.Tensor(force[i].reshape(-1, n_atom, 3)).to(device)
            f = torch.matmul(CG, f0)
            mean_force = f.pow(2).sum(2).mean()

            loss_fm = mean_force

            if epoch >= args.fm_epoch:
                loss = loss_ae + args.fm_co * loss_fm
            else:
                loss = loss_ae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            loss_ae_epoch += loss_ae.item()
            loss_fm_epoch += loss_fm.item()

        loss_epoch = loss_epoch / xyz.shape[0]
        loss_ae_epoch = loss_ae_epoch / xyz.shape[0]
        loss_fm_epoch = loss_fm_epoch / xyz.shape[0]

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
