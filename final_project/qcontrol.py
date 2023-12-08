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
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, **kwargs):
        """
        Initialize the environment

        Args:
            td_params (_type_, optional): _description_. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to "cpu".
            kwargs: must include PSI_0, PSI_F, H0, H1, complex_type
        """
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])

        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64, device=device).random_().item()
        self.set_seed(seed)

        # Unpack the kwargs
        # global device
        # device = device_
        # global complex_type
        # complex_type = kwargs.get("complex_type", torch.complex64)
        # global PSI_0
        # PSI_0 = kwargs.get("PSI_0")
        # global PSI_0_tensor
        # PSI_0_tensor = torch.tensor(PSI_0, dtype=complex_type).to(device)
        # self.PSI_0_tensor = PSI_0_tensor
        # global PSI_F
        # PSI_F = kwargs.get("PSI_F")
        # global PSI_F_tensor
        # PSI_F_tensor = torch.tensor(PSI_F, dtype=complex_type).to(device)
        # global H0
        # H0 = kwargs.get("H0")
        # global H0_tensor
        # H0_tensor = torch.tensor(H0, dtype=complex_type).to(device)
        # global H1
        # H1 = kwargs.get("H1")
        # global H1_tensor
        # H1_tensor = torch.tensor(H1, dtype=complex_type).to(device)

    # Helpers: _make_step and gen_params
    @staticmethod
    def gen_params(batch_size=None, **kwargs) -> TensorDictBase:
        """Returns a tensordict containing the physical parameters such as timestep and control stuff."""
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

    # Specifies the bounds of the environment
    def _make_spec(self, td_params):

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
            # we need to add the "params" to the observation specs, as we want
            # to pass it at each step during a rollout
            # params=make_composite_from_td(td_params["params"]),
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
            low=-1,
            high=1,
            shape=(1,),
            dtype=torch.float32,
            device=device
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    # Reset function resets a run to the original starting state
    def _reset(self, tensordict):


        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)
        # else:
        #     psi_real = psi_real.repeat(tensordict.size(dim=0), 1, 1)
        #     psi_imag = psi_imag.repeat(tensordict.size(dim=0), 1, 1)
        psi_real = torch.real(tensordict["params", "psi_0"])
        psi_imag = torch.imag(tensordict["params", "psi_0"])

        out = TensorDict(
            {
                "psi_real": psi_real,
                "psi_imag": psi_imag,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )

        # print("out psi_real", out["psi_real"])
        return out

    # Step function advances along the environment
    @staticmethod
    def _step(tensordict):


        # Current quantum state and control value
        psi_real, psi_imag, control = tensordict["psi_real"], tensordict["psi_imag"], tensordict["action"].squeeze(-1)
        psi = psi_real + 1.0j*psi_imag

        # Target final states and Hamiltonians
        psi_f = tensordict["params", "psi_f"]
        H0 = tensordict["params", "H0"]
        H1 = tensordict["params", "H1"]

        # Timestep and other parameters
        dt = tensordict["params", "dt"]
        # step = tensordict["params", "step"]
        # t = dt*step
        eps = tensordict["params", "eps"]
        control_thresh = tensordict["params", "control_thresh"]
        control_penalty = tensordict["params", "control_penalty"]

        # Cost Function -- Log Infidelity with Final State 1 - F
        # print("psi", psi)
        # print("PSI_F_tensor", PSI_F_tensor)
        # costs = 1 - torch.pow(torch.abs(torch.transpose(torch.conj_physical(PSI_F_tensor), 0, 1)@psi), 2)
        bra_ket = torch.sum(torch.conj_physical(psi_f)*psi, dim=[-2])
        F = torch.pow(torch.abs(bra_ket), 2).squeeze()
        # breakpoint()

        # If you're super close to the state STOP
        done = F >= tensordict["params", "close_thresh"]

        # Add penalty for large control
        control_term = control_penalty*torch.clamp(torch.pow(torch.abs(control), 2) - control_thresh, min=0).squeeze()

        # Maybe add some sort of target gate time and give a penalty,
        # terminate past the gate
        # Time and be more forgiving for being far away earlier
        # time_term = torch.pow(t, 2)
        
        # breakpoint()
        costs = -torch.log(F + eps) + control_term
        reward = -costs.view(*tensordict.shape, 1)

        # Propagate along the state
        H_dim = H0.shape[-2]*H0.shape[-1]
        control_reshape = control.repeat(1,H_dim,1).transpose(0,2).reshape(H0.shape)
        dt_reshape = dt.repeat(1,H_dim,1).transpose(0,2).reshape(H0.shape)
        # print(control_reshape)


        U = torch.linalg.matrix_exp(-1.0j*dt_reshape*(H0 + control_reshape*H1))
        new_psi = U@psi

        # breakpoint()

        # print(torch.abs(new_psi[0]), new_psi[0], control[0])

        # Set done if you're under a threshold of closeness -- 99.9%

        # done = torch.zeros_like(reward, dtype=torch.bool, device=device)
        out = TensorDict(
            {
                "psi_real": torch.real(new_psi),
                "psi_imag": torch.imag(new_psi),
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
                # "step": tensordict["step"] + 1,
            },
            tensordict.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

def make_composite_from_td(td):
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


if __name__ == "__main__":

    env = QuantumEnv()

    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)

    # print(env.reset())    

    # Test the environment
    # check_env_specs(env)

    # Make the network
    torch.manual_seed(0)
    env.set_seed(0)

    class myNet(nn.Module):

        def __init__(self):
            super(myNet, self).__init__()
            ctype = torch.float32
            self.net = nn.Sequential(
                        nn.Linear(4, 64, dtype=ctype),
                        nn.Tanh(),
                        nn.Linear(64, 32, dtype=ctype),
                        nn.Tanh(),
                        nn.Linear(32, 16, dtype=ctype),
                        nn.Tanh(),
                        nn.Linear(16, 1, dtype=ctype),
                        # nn.Tanh()
                        )

        def forward(self, x1, x2):
            # print(x1.shape, x2.shape)
            # flattened = torch.concatenate((x1.flatten(), x2.flatten()))
            # flattened = 1.0j*torch.hstack((x1, x2)).squeeze()
            flattened = torch.hstack((x1, x2)).squeeze()
            return self.net(flattened)

    net = myNet().to(device)
    policy = TensorDictModule(
        net,
        in_keys=["psi_real", "psi_imag"],
        out_keys=["action"]
    )
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)

    # Train Parameters
    batch_size = 2
    rollout_len = 1500
    total_trials = 10000
    pbar = tqdm.tqdm(range(total_trials // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_trials)
    logs = defaultdict(list)


    # System Parameters
    dt = 0.005
    gate_time = 10
    control_thresh = 0.1
    control_penalty = 1
    np_ctype = np.complex64
    psi_0 = np.array([1, 0]).reshape((2,1)).astype(np_ctype)
    psi_f = np.array([0, 1]).reshape((2,1)).astype(np_ctype)
    H0 = np.array([[1,  0],
                   [0, -1]]).astype(np_ctype)
    H1 = np.array([[0,  1],
                   [1,  0]]).astype(np_ctype)
    params = env.gen_params(batch_size=[batch_size], dt=dt,
                            gate_time=gate_time, control_thresh=control_thresh,
                            control_penalty=control_penalty, psi_0=psi_0,
                            psi_f=psi_f, H0=H0, H1=H1)

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
        scheduler.step()

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

    dt = 0.005
    timesteps = 2000
    tlist = np.arange(timesteps)*dt

    def get_control_pulse(net, timesteps=timesteps, dt=dt, ctype=torch.complex64):
        psi = torch.tensor(psi_0, dtype=ctype, device=device)
        H0_tensor = torch.tensor(H0, dtype=ctype, device=device)
        H1_tensor = torch.tensor(H1, dtype=ctype, device=device)
        cntrl_seq = torch.zeros(timesteps, dtype=ctype)
        psi_list = []
        for i in range(timesteps):
            cntrl = net(torch.real(psi).reshape((1,2,1)), torch.imag(psi).reshape((1,2,1)))
            cntrl_seq[i] = cntrl
            U = torch.linalg.matrix_exp(-1.0j*dt*(H0_tensor+cntrl*H1_tensor))
            psi = U@psi
            psi_list.append(psi.cpu().detach().numpy())

        return cntrl_seq.cpu().detach().numpy(), psi_list

    pulse, psi_list = get_control_pulse(net)

    def Ham(t, tlist):
        tdiff = np.abs(tlist - t)
        return qt.Qobj(H0 + pulse[np.argmin(tdiff)]*H1)

    states_qt = qt.sesolve(lambda t, args: Ham(t, tlist), qt.Qobj(psi_0), tlist=tlist)
    prob0_qt = [np.abs(s[0][0][0])**2 for s in states_qt.states]
    prob0_prop = [np.abs(s[0])**2 for s in psi_list]

    f, ax = plt.subplots(ncols=2)
    ax[0].set_title("pulse")
    ax[1].set_title("population")
    ax[0].plot(tlist, np.abs(pulse))
    ax[1].plot(tlist, prob0_qt, label="qutip")
    ax[1].plot(tlist, prob0_prop, label="prop")
    ax[1].legend()
    plt.savefig("pulse.png")
