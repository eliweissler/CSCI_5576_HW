from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
import qutip as qt

from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.utils import check_env_specs, step_mdp

global device
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
# device = "cpu"
print("Using device", device)


import matplotlib
from matplotlib import pyplot as plt

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class QuantumEnv(EnvBase):
    """
    Environment for doing reinforcement learning with
    schrodinger dynamics

    Mandatory Methods:
     _step
     _reset
     _set_seed

    Helpers:
     gen_params
     _make_spec
     make_composite_from_td

    """
    metadata = {}
    batch_locked = False

    def __init__(self, td_params=None, seed=None, **kwargs):
        """
        Initialize the environment

        Args:
            td_params (Tensordict, optional): Tensordict with parameters for the run.
            seed (int, optional): Random seed for reproducability. Defaults to None.
        """
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])

        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=device).random_().item()
        self.set_seed(seed)

    @staticmethod
    def gen_params(batch_size=None, **kwargs) -> TensorDictBase:
        """
        Returns a tensordict containing the physical parameters
        such as timestep and initial and final states that
        define the run.

        Args:
            batch_size (int, optional): batch size.
            kwargs:
                dt: timestep
                close_thresh: Fidlity threshold at which to
                              consider the final state reached.
                control_penalty: coefficient in front of the quadratic penalty
                                 in the cost function.
                control_thresh: NOT CURRENTLY USED
                gate_time: target gate time
                psi_0: matrix with columns representing the starting states
                psi_f: matrix with columns representing the ending states
                H0: Hamiltonian of the system without a drive
                H1: Hamiltonian of the drive

        Returns:
            TensorDictBase: tensordict with these parameters
        """
        if batch_size is None:
            batch_size = []
        ctype = kwargs.get("complex_type", torch.complex64)
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "dt": kwargs.get("dt", 0.005),
                        "close_thresh": kwargs.get("close_thresh", 0.9999),
                        "eps": kwargs.get("eps", 1e-10),
                        "control_penalty": kwargs.get("control_penalty", 20),
                        "control_thresh": kwargs.get("control_thresh", 0.1),
                        "gate_time": kwargs.get("gate_time", 10),
                        "psi_0": kwargs.get("psi_0", torch.tensor(np.array([0, 1]).reshape((2, 1)), dtype=ctype)),
                        "psi_f": kwargs.get("psi_f", torch.tensor(np.array([1, 0]).reshape((2, 1)), dtype=ctype)),
                        "H0": kwargs.get("H0", torch.tensor(np.array([[1,  0], [0, -1]]), dtype=ctype)),
                        "H1": kwargs.get("H1", torch.tensor(np.array([[0,  1], [1, 0]]), dtype=ctype))
                    },
                    [],
                )
            },
            [],
            device=device
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params):
        """
        Defines the bounds for the environment and
        control variables

        Args:
            td_params (TensorDict): parameters for the system
        """

        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
            psi_real=BoundedTensorSpec(
                low=-1,
                high=1,
                shape=td_params["params", "psi_0"].shape,
                device=device,
                dtype=torch.float32),
            psi_imag=BoundedTensorSpec(
                low=-1,
                high=1,
                shape=td_params["params", "psi_0"].shape,
                device=device,
                dtype=torch.float32),
            t_left=BoundedTensorSpec(
                low=0,
                high=1,
                shape=[],
                device=device,
                dtype=torch.float32),
            # we need to add the "params" to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
            device=device
        )

        # since the environment is stateless, we expect the previous output as input.
        # For this, EnvBase expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
            low=-td_params["params", "control_thresh"],
            high=td_params["params", "control_thresh"],
            shape=(1,),
            dtype=torch.float32,
            device=device
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def _reset(self, tensordict):
        """
        Resets the environment to the original starting state

        Args:
            tensordict (TensorDict): the environment's current tensordict

        Returns:
            TensorDict: tensordict describing the reset environment
        """

        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        psi_0 = tensordict["params", "psi_0"]
        t_left = tensordict["params", "gate_time"]/tensordict["params", "gate_time"]

        psi_real = torch.real(psi_0)
        psi_imag = torch.imag(psi_0)

        out = TensorDict(
            {
                "psi_real": psi_real,
                "psi_imag": psi_imag,
                "t_left": t_left,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )

        return out
    
    @staticmethod
    def avg_fidelity(psi_f, psi):
        """
        Helper function to calculate the average end state fidelity

        Args:
            psif (torch.tensor complex): _description_
            psi (torch.tensor complex): _description_

        Returns:
            _type_: _description_
        """

        bra_ket = torch.sum(torch.conj_physical(psi_f)*psi, dim=[-2])
        F = torch.mean(torch.pow(torch.abs(bra_ket), 2).squeeze(),dim=-1)

        return F

    @staticmethod
    def _step(tensordict):
        """
        Advances the enviroment according to the schrodinger equation

        Args:
            tensordict (TensorDict): tensordict that describes the environment

        Returns:
            TensorDict: tensordict for the updated environment
        """


        # Current quantum state
        psi_real, psi_imag = tensordict["psi_real"], tensordict["psi_imag"]
        psi = psi_real + 1.0j*psi_imag

        # Control Value
        control = tensordict["action"].squeeze(-1)

        # Time left (as a decimal between 0-1) and gate time (units)
        t_left = tensordict["t_left"]
        gate_time = tensordict["params", "gate_time"]

        # Target final states and Hamiltonians
        psi_f = tensordict["params", "psi_f"]
        H0 = tensordict["params", "H0"]
        H1 = tensordict["params", "H1"]

        # Timestep and other parameters
        dt = tensordict["params", "dt"]
        # step = tensordict["params", "step"]
        # t = dt*step
        eps = tensordict["params", "eps"]
        # control_thresh = tensordict["params", "control_thresh"]
        control_penalty = tensordict["params", "control_penalty"]

        
        # Clip control
        # control = torch.clamp(control, min=-control_thresh, max=control_thresh)

        # Cost Function -- Log Infidelity with Final State 1 - F
        # print("psi", psi)
        # print("PSI_F_tensor", PSI_F_tensor)
        # costs = 1 - torch.pow(torch.abs(torch.transpose(torch.conj_physical(PSI_F_tensor), 0, 1)@psi), 2)
        # bra_ket = torch.sum(torch.conj_physical(psi_f)*psi, dim=[-2])
        # F = torch.mean(torch.pow(torch.abs(bra_ket), 2).squeeze(),dim=-1)

        F = QuantumEnv.avg_fidelity(psi_f, psi)

        # If you're super close to the state STOP
        done = F >= tensordict["params", "close_thresh"]

        # Add penalty for large control
        # control_time_env = 1 - torch.exp(-torch.pow((0.5-t_left)/0.2, 2))
        control_time_env = 1
        control_cost = control_time_env*control_penalty*torch.pow(control, 2)

        # Maybe add some sort of target gate time and give a penalty,
        # terminate past the gate
        # Time and be more forgiving for being far away earlier
        # time_term = torch.pow(t, 2)
        costs = -torch.pow((1-t_left), 2)*torch.log(F + eps) + control_cost
        reward = -costs.view(*tensordict.shape, 1)

        # Propagate along the state
        H_dim = H0.shape[-2]*H0.shape[-1]
        control_reshape = control.repeat(1, H_dim, 1).transpose(0, 2).reshape(H0.shape)
        dt_reshape = dt.repeat(1, H_dim, 1).transpose(0, 2).reshape(H0.shape)


        U = torch.linalg.matrix_exp(-1.0j*dt_reshape*(H0 + control_reshape*H1))
        new_psi = U@psi

        out = TensorDict(
            {
                "psi_real": torch.real(new_psi),
                "psi_imag": torch.imag(new_psi),
                "t_left": t_left - dt/gate_time,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        """
        Sets the torch random seed

        Args:
            seed (Optional[int]): integer seed
        """
        rng = torch.manual_seed(seed)
        self.rng = rng

def make_composite_from_td(td):
    """
    Helper function to create a tensordict of unbounded values
    """
    # custom funtion to convert a tensordict in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


def get_control_pulse(net, timesteps, dt, ctype=torch.complex64):
    """
    Returns a control pulse and state history for a trained agent
    network

    Args:
        net (_type_): _description_
        timesteps (_type_): _description_
        dt (_type_): _description_
        ctype (_type_, optional): _description_. Defaults to torch.complex64.

    Returns:
        _type_: _description_
    """
    # Put the network into eval mode (no gradients)
    net.eval()

    # Make the initial state tensors
    psi = torch.tensor(psi_0, dtype=ctype, device=device)
    t_left = torch.tensor(1, device=device)
    t_increment = torch.tensor(dt/gate_time, device=device)
    H0_tensor = torch.tensor(H0, dtype=ctype, device=device)
    H1_tensor = torch.tensor(H1, dtype=ctype, device=device)
    cntrl_seq = torch.zeros(timesteps, dtype=ctype)
    psi_list = []
    for i in range(timesteps):
        cntrl = net(torch.real(psi).reshape((1,4,1)), torch.imag(psi).reshape((1,4,1)), t_left)
        cntrl_seq[i] = cntrl
        U = torch.linalg.matrix_exp(-1.0j*dt*(H0_tensor+cntrl*H1_tensor))
        psi = U@psi
        psi_list.append(psi.cpu().detach().numpy())
        t_left = t_left - t_increment

    return cntrl_seq.cpu().detach().numpy(), psi_list


if __name__ == "__main__":

    dt = 0.01
    gate_time = 30
    control_thresh = 1
    control_penalty = 15
    close_thresh = 0.99999

    np_ctype = np.complex64
    psi_0 = np.array([[1,  0],
                        [0,  1]]).astype(np_ctype)
    psi_f = np.array([[0,  1],
                        [1,  0]]).astype(np_ctype)
    H0 = np.array([[1,  0],
                    [0, -1]]).astype(np_ctype)
    H1 = np.array([[0,  1],
                [1,  0]]).astype(np_ctype)


    env = QuantumEnv()
    check_env_specs(env)

    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)

    # print(env.reset())    

    # Test the environment

    # Make the network
    torch.manual_seed(0)
    env.set_seed(0)

    class myNet(nn.Module):

        def __init__(self):
            super(myNet, self).__init__()
            ctype = torch.float32
            p_drop = 0.01
            bias = True
            self.net = nn.Sequential(
                        nn.Linear(2*psi_0.size + 1, 64, dtype=ctype, bias=bias),
                        nn.Tanh(),
                        # nn.Dropout(p=p_drop),
                        nn.Linear(64, 64, dtype=ctype, bias=bias),
                        nn.Tanh(),
                        nn.Dropout(p=p_drop),
                        nn.Linear(64, 16, dtype=ctype, bias=bias),
                        nn.Tanh(),
                        nn.Dropout(p=p_drop),
                        nn.Linear(16, 1, dtype=ctype, bias=bias),
                        # nn.Tanh()
                        )

        def forward(self, x1, x2, x3):
            # print(x1.shape, x2.shape)
            # flattened = torch.concatenate((x1.flatten(), x2.flatten()))
            # flattened = 1.0j*torch.hstack((x1, x2)).squeeze()
            flattened = torch.hstack((x1.flatten(1, -1), x2.flatten(1, -1), x3.reshape(-1,1)))
            # ctype = torch.float32
            # self.net = nn.Sequential(
            #             nn.Linear(4, 64, dtype=ctype),
            #             nn.Tanh(),
            #             nn.Linear(64, 32, dtype=ctype),
            #             nn.Tanh(),
            #             nn.Linear(32, 16, dtype=ctype),
            #             nn.Tanh(),
            #             nn.Linear(16, 1, dtype=ctype),
            #             # nn.Tanh()
            #             )
            # x = torch.hstack((x1, x2)).squeeze()
            # x = nn.Linear(4, 64, dtype=ctype)(x)
            # x = nn.Tanh(x)
            # x = torch.clamp(self.net(flattened), min=-control_thresh, max=control_thresh)
            return self.net(flattened)
        # @staticmethod
        # def dropout_complex(x):
        #     # work around unimplemented dropout for complex
        #     if x.is_complex():
        #         mask = torch.nn.functional.dropout(torch.ones_like(x.real))
        #         return x * mask
        #     else:
        #         return torch.nn.functional.dropout(x)

    net = myNet().to(device)
    policy = TensorDictModule(
        net,
        in_keys=["psi_real", "psi_imag", "t_left"],
        out_keys=["action"]
    )
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

    # Train Parameters
    batch_size = 100
    rollout_len = int(gate_time/dt)
    total_trials = 10000//2
    # total_trials = 50
    pbar = tqdm.tqdm(range(total_trials // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_trials)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optim, total_iters=total_trials)
    logs = defaultdict(list)

    # Keep track of the best run throughout training
    logs["best_run"] = {}
    logs["best_run"]["fidelity"] = 0
    logs["best_run"]["reward"] = -np.inf
    logs["best_run"]["control"] = None
    logs["best_run"]["psi"] = None

    params = env.gen_params(batch_size=[batch_size], dt=dt,
                            gate_time=gate_time, control_thresh=control_thresh,
                            control_penalty=control_penalty, psi_0=psi_0,
                            psi_f=psi_f, H0=H0, H1=H1, close_thresh=close_thresh)
    for _ in pbar:
        init_td = env.reset(params)
        rollout = env.rollout(rollout_len, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())

        psi_final = rollout["psi_real"][:, -1] + 1.0j*rollout["psi_imag"][:, -1]
        F_final = QuantumEnv.avg_fidelity(params["params","psi_f"], psi_final)
        ibest = torch.argmax(F_final)
        rbest = rollout["next", "reward"][ibest, -1]
        fbest = F_final[ibest]
        if fbest > logs["best_run"]["fidelity"]:
            logs["best_run"]["reward"] = rbest
            logs["best_run"]["psi"] = rollout["psi_real"][ibest] + 1.0j*rollout["psi_imag"][ibest]
            logs["best_run"]["control"] = rollout["action"][ibest]
            logs["best_run"]["fidelity"] = fbest

        scheduler.step()

    timesteps = rollout_len
    tlist = np.arange(timesteps)*dt

    pulse = logs["best_run"]["control"].cpu().detach().numpy()
    psi_hist = logs["best_run"]["psi"].cpu().detach().numpy()
    fid = logs["best_run"]["fidelity"].cpu().detach().numpy()
    len_pulse = pulse.size

    prob0_start0 = [np.abs(s[0, 0])**2 for s in psi_hist]
    prob0_start1 = [np.abs(s[0, 1])**2 for s in psi_hist]

    f, ax = plt.subplots(ncols=2)
    f.set_size_inches((8, 3))
    ax[0].set_title(f"pulse (best) - fidelity: {str(np.round(fid, 6))}")
    ax[1].set_title("population in [1,0] state")
    ax[0].plot(tlist[:len_pulse], pulse)
    ax[1].plot(tlist[:len_pulse], prob0_start0, label="starts in [1,0]")
    ax[1].plot(tlist[:len_pulse], prob0_start1, label="starts in [0,1]")
    ax[1].legend()
    plt.savefig("pulse_best.png")


    def plot():
        with plt.ion():
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(logs["return"])
            plt.title("returns")
            plt.xlabel("iteration")
            plt.subplot(1, 2, 2)
            plt.plot(logs["last_reward"])
            plt.title("last reward")
            plt.xlabel("iteration")
            if is_ipython:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            plt.savefig("training.png")

    plot()


    def get_control_pulse(net, params, timesteps=timesteps, dt=dt, ctype=torch.complex64):

        net.eval()
        psi = params["params"]["psi_0"][0] # torch.tensor(psi_0, dtype=ctype, device=device)
        t_left = torch.tensor(1, device=device)
        t_increment = params["params"]["dt"][0]/params["params"]["gate_time"][0]
        H0_tensor = params["params"]["H0"][0]#torch.tensor(H0, dtype=ctype, device=device)
        H1_tensor = params["params"]["H1"][0]#torch.tensor(H1, dtype=ctype, device=device)
        psi_f = params["params"]["psi_f"][0]
        cntrl_seq = torch.zeros(timesteps, dtype=ctype, device=device)
        fid = torch.zeros(timesteps, device=device)
        psi_list = []
        for i in range(timesteps):
            cntrl = net(torch.real(psi).reshape((1,4,1)), torch.imag(psi).reshape((1,4,1)), t_left)
            cntrl_seq[i] = cntrl
            U = torch.linalg.matrix_exp(-1.0j*dt*(H0_tensor+cntrl*H1_tensor))
            psi = U@psi
            fid[i] = QuantumEnv.avg_fidelity(psi_f, psi)
            psi_list.append(psi.cpu().detach().numpy())
            t_left = t_left - t_increment

        return cntrl_seq.cpu().detach().numpy(), psi_list, fid.cpu().detach().numpy()

    pulse, psi_list, fid = get_control_pulse(net, params)

    def Ham(t, tlist):
        tdiff = np.abs(tlist - t)
        return qt.Qobj(H0 + pulse[np.argmin(tdiff)]*H1)

    # states_qt = qt.sesolve(lambda t, args: Ham(t, tlist), qt.Qobj(psi_0), tlist=tlist)
    # prob0_qt = [np.abs(s[0][0][0])**2 for s in states_qt.states]

    prob0_start0 = [np.abs(s[0,0])**2 for s in psi_list]
    prob0_start1 = [np.abs(s[0,1])**2 for s in psi_list]

    f, ax = plt.subplots(ncols=2)
    f.set_size_inches((8, 3))
    ax[0].set_title(f"pulse (from network) - F: {str(np.round(fid[-1], 6))}")
    ax[1].set_title("population in [1,0] state")
    ax[0].plot(tlist, pulse)
    # ax[1].plot(tlist, prob0_qt, label="qutip")
    ax[1].plot(tlist, prob0_start0, label="starts in [1,0]")
    ax[1].plot(tlist, prob0_start1, label="starts in [0,1]")
    ax[1].legend()
    plt.savefig("pulse.png")
