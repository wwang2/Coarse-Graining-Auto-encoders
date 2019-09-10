import mdtraj as md
import numpy as np


def load_data(args):
    otp = md.load("data/otp.pdb")
    otp_top = otp.top.to_dataframe()[0]
    otp_element = otp_top['element'].values.tolist()
    xyz = np.load('data/otp_xyz.npy')
    force = np.load('data/otp_force.npy')
    xyz = xyz[-args.ntr:] * 10
    force = force[-args.ntr:] * 0.0239
    return xyz, force, otp_element
