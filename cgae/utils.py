import numpy as np


def save_traj(traj, Z, name):
    """Summary
    
    Args:
        traj (np.array): traj
        Z (atomic number): Description
        name (filename): Description
    """
    traj = np.array(traj)
    Z = np.array(Z * traj.shape[0]).reshape(traj.shape[0], len(Z), 1)
    traj_write = np.dstack((Z, traj))
    write_traj(filename=name, frames=traj_write)


def write_traj(filename, frames):
    '''
        Write trajectory dataframes into .xyz format for VMD visualization
        to do: include multiple atom types 
        
        example:
            path = "../../sim/topotools_ethane/ethane-nvt_unwrap.xyz"
            traj2write = trajconv(n_mol, n_atom, box_len, path)
            write_traj(path, traj2write)
    '''
    file = open(filename, 'w')
    atom_no = frames.shape[1]
    for i, frame in enumerate(frames):
        file.write(str(atom_no) + '\n')
        file.write('Atoms. Timestep: ' + str(i) + '\n')
        for atom in frame:
            if atom.shape[0] == 4:
                try:
                    file.write(str(int(atom[0])) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
                except:
                    file.write(str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write("1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()
