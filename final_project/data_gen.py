import h5py

import numpy as np
import pandas as pd
import qutip as qt
import scipy as sp

from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def read_data(hfile: str, dim: int, label: str, i_start: int, i_end: int):
    """
    Args:
        hfile (str): _description_
        dim (int): _description_
        label (str): _description_
        i_start (int): _description_
        i_end (int): _description_

    Returns:
        np.array, np.array : entries from the hdf5 file with shape
                            (n_entries, n_timesteps, dim) -- states
                            (n_entries, n_timesteps, dim, dim) -- H
    """

    with h5py.File(hfile, "r") as f:
        state_re = f[f"dim_{dim}/{label}_state_re"][i_start:i_end]
        state_im = f[f"dim_{dim}/{label}_state_im"][i_start:i_end]
        H_re = f[f"dim_{dim}/{label}_H_re"][i_start:i_end]
        H_im = f[f"dim_{dim}/{label}_H_im"][i_start:i_end]

    states = np.zeros_like(state_re, dtype=np.complex128)
    states += state_re
    states += 1.0j*state_im

    H = np.zeros_like(H_re, dtype=np.complex128)
    H += H_re
    H += 1.0j*H_im

    return states, H


def dset_size(hfile: str, dim: int, label: str):
    """
    Args:
        hfile (str): _description_
        dim (int): _description_
        label (str): _description_

    Returns:
        shape of dataset
    """

    with h5py.File(hfile, "r") as f:
        shape = f[f"dim_{dim}/{label}_state_re"].shape

    return shape


def gen_data(hfile: str, dim: int, n_samples: int, label: str, batchsize: int = 50,
             t_end: float = 10, dt: float = 0.1, vary_time: float = 0):

    # Timesteps to solve for
    # Choose a random time up to t_end to end the pulse
    # tlist = np.arange(0, np.random.random()*t_end+dt, dt)
    tlist = np.arange(0, t_end+dt, dt)
    n_steps = tlist.size

    mode = "w"
    if Path(hfile).exists():
        mode = 'a'

    with h5py.File(hfile, mode) as f:

        # Make group and save attributes
        grp_name = f"dim_{dim}"
        if grp_name not in [x for x in f]:
            grp = f.create_group(grp_name)
        else:
            grp = f[grp_name]

        # Make datasets
        if f"{label}_state_re" in [x for x in grp]:
            del grp[f"{label}_state_re"]
            del grp[f"{label}_state_im"]
            del grp[f"{label}_H_re"]
            del grp[f"{label}_H_im"]

        dset_re = grp.create_dataset(f"{label}_state_re", (n_samples, n_steps, dim), dtype="f")
        dset_im = grp.create_dataset(f"{label}_state_im", (n_samples, n_steps, dim), dtype="f")
        H_re = grp.create_dataset(f"{label}_H_re", (n_samples, n_steps, dim, dim), dtype="f")
        H_im = grp.create_dataset(f"{label}_H_im", (n_samples, n_steps, dim, dim), dtype="f")

        # Add metadata
        dset_re.attrs["t_end"] = t_end
        dset_re.attrs["dt"] = dt
        dset_im.attrs["t_end"] = t_end
        dset_im.attrs["dt"] = dt

        print("Generating data for dimension", dim, "for file", hfile, "label:", label)
        to_save_states = np.zeros((batchsize, n_steps, dim),
                                  dtype=np.complex128)
        to_save_H = np.zeros((batchsize, n_steps, dim, dim),
                             dtype=np.complex128)
        for i in tqdm(range(n_samples)):

            state = qt.rand_ket(dim)
            if vary_time > 0:

                # Make a random constant part that's diagonal
                H0 = random_ham(dim, diag_real=True)

                # Make a random oscillating part that's
                # up to the size of vary_time
                H1 = np.random.random()*vary_time*random_ham(dim)
                h1_coeff = random_func(tlist, max_freq=(np.abs(H1.full()).max() -
                                                        np.abs(H1.full()).min())*2)

                Hlist = [H0, [H1, lambda t, args: h1_coeff(t)]]
                # Hlist = [[H0, 1]]
                res = qt.sesolve(Hlist, state, tlist)

                hams = np.array([(H0 + H1*h1_coeff(t)).full() for t in tlist])
                to_save_H[i % batchsize] = hams.reshape(n_steps, dim, dim)

            else:
                # Make a random constant part
                H = random_ham(dim)
                res = qt.sesolve(H, state, tlist)
                to_save_H[i % batchsize] = np.array([H.full() for s in res.states]).reshape(n_steps, dim, dim)

            to_save_states[i % batchsize] = np.array([s.full() for s in res.states]).reshape(n_steps, dim)

            # At the end of a batch
            if (i % batchsize == batchsize - 1):
                dset_re[i-(batchsize-1):i+1] = np.real(to_save_states)
                dset_im[i-(batchsize-1):i+1] = np.imag(to_save_states)
                H_re[i-(batchsize-1):i+1] = np.real(to_save_H)
                H_im[i-(batchsize-1):i+1] = np.imag(to_save_H)

            # At the end of the dataset
            elif i >= n_samples - 1:
                dset_re[i-(i % batchsize):i+1] = np.real(to_save_states[:i % batchsize + 1])
                dset_im[i-(i % batchsize):i+1] = np.imag(to_save_states[:i % batchsize + 1])
                H_re[i-(i % batchsize):i+1] = np.real(to_save_H[:i % batchsize + 1])
                H_im[i-(i % batchsize):i+1] = np.imag(to_save_H[:i % batchsize + 1])

    return


def random_ham(dim: int, diag_real: bool = False, scale: float = 1):
    """
    Generates a random hamiltonian using qt.rand_herm. By default
    magnitudes of all numbers are between 0-1. Can adjust using scale.

    Args:
        dim (int): Dimension of system
        diag_real (bool, optional): Whether you want it diagonal + real or not.
        scale (float, optional): Overall multiplier. Defaults to 1.

    Returns:
        qt.Qobj: the random Hamiltonian
    """

    # Generate random Hamiltonian
    if diag_real:
        H = scale*qt.Qobj(np.diag(np.random.random(dim)*2 - 1))

    # Generate random Hamiltonian... Might be diagonal
    else:
        H = scale*qt.rand_herm(dim)

    return H


def gaussian_envelope(t: float, tg: float) -> float:
    """
    Generates a Gaussian pulse envelope,
    starting a t = 0 and going until t = tg,
    peaking with an envelope value of 1 at halfway through

    Args:
        t (float): time
        tg (float): total gate time (0 at t = 0 and t = tg)
        omega (float): freq of cos drive
        A (float): height of Gaussian

    Returns:
        float: drive value at time t
    """
    envelope = np.exp(-8 * t * (t - tg) / tg**2)
    envelope /= np.exp(-8 * (tg/2) * (tg/2 - tg) / tg**2)
    envelope += -envelope[0]
    envelope /= np.max(envelope)
    return envelope


def random_func(t: np.array, max_freq: int, envelope_fn=gaussian_envelope):
    """
    A randomly generated continous function of time at the specified
    timesteps

    Args:
        t (np.array): timesteps
        envelope_fn (callable, optional): overall envelope function.
    """

    y = (np.random.random()*np.sin(np.random.random()*max_freq*t) +
         np.random.random()*np.cos(np.random.random()*max_freq*t))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    # Multiply by envelope function
    return sp.interpolate.interp1d(t, y*envelope_fn(t, t[-1]), kind="linear",
                                   bounds_error=False, fill_value=0)






# TODO: Random state, random hamiltonian. Predict the n+1 timestep


if __name__ == "__main__":

    hfile = "data.hdf5"
    gen_data(hfile, 2, 1000, "train", vary_time=1)
    gen_data(hfile, 2, 100, "test", vary_time=1)