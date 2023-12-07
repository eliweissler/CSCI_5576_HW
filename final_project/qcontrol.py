from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm

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
print("Using device", device)
# device = "cpu"

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
        global complex_type
        complex_type = kwargs.get("complex_type", torch.complex64)
        global PSI_0
        PSI_0 = kwargs.get("PSI_0")
        global PSI_0_tensor
        PSI_0_tensor = torch.tensor(PSI_0, dtype=complex_type).to(device)
        self.PSI_0_tensor = PSI_0_tensor
        global PSI_F
        PSI_F = kwargs.get("PSI_F")
        global PSI_F_tensor
        PSI_F_tensor = torch.tensor(PSI_F, dtype=complex_type).to(device)
        global H0
        H0 = kwargs.get("H0")
        global H0_tensor
        H0_tensor = torch.tensor(H0, dtype=complex_type).to(device)
        global H1
        H1 = kwargs.get("H1")
        global H1_tensor
        H1_tensor = torch.tensor(H1, dtype=complex_type).to(device)

    # Helpers: _make_step and gen_params
    @staticmethod
    def gen_params(dt=0.01, batch_size=None) -> TensorDictBase:
        """Returns a tensordict containing the physical parameters such as timestep and control stuff."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "dt": dt
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
                shape=[2,1],
                device=device,
                dtype=torch.float32),
            psi_imag=BoundedTensorSpec(
                low=-1,
                high=1,
                shape=[2,1],
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

        # print("-----------------------")
        # print("in reset")

        psi_real = torch.real(PSI_0_tensor)
        psi_imag = torch.imag(PSI_0_tensor)

        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)
        else:
            psi_real = psi_real.repeat(tensordict.size(dim=0), 1, 1)
            psi_imag = psi_imag.repeat(tensordict.size(dim=0), 1, 1)


        # print("batch_dims", self.batch_dims)
            
        # print("psi_real - ", self.batch_dims, psi_real.shape)

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

        # print("------------------")
        # print("in step")

        # Current quantum state and control value
        psi_real, psi_imag, control = tensordict["psi_real"], tensordict["psi_imag"], tensordict["action"].squeeze(-1)
        # print("psi_re", psi_real)
        # print("psi_im", psi_imag)
        # print("control", control)
        psi = psi_real + 1.0j*psi_imag

        # Timestep
        dt = tensordict["params", "dt"]

        # Cost Function -- Fidelity with Final State
        # print("psi", psi)
        # print("PSI_F_tensor", PSI_F_tensor)
        costs = 1 - torch.pow(torch.abs(torch.transpose(torch.conj_physical(PSI_F_tensor), 0, 1)@psi), 2)
        # Add penalty for large control
        max_val = 0.1
        multiplier = 1
        costs += multiplier*torch.clamp(torch.abs(control) - max_val, min=0).reshape(costs.shape)

        # Propagate along the state
        # print("H0", H0_tensor)
        # print("H1", H1_tensor)
        # print("control.shape", control.shape)
        # print("tensordict shape", tensordict.shape)
        if len(control.shape) > 0:
            H_dim = H1_tensor.size(dim=0)
            c_dim = control.size(dim=0)
            control_repeat = control.repeat(1,H_dim**2,1).T.reshape(c_dim, H_dim, H_dim)
            dt_repeat = dt.repeat(1,H_dim**2,1).T.reshape(c_dim, H_dim, H_dim)
            H0_repeat = H0_tensor.repeat(c_dim,1,1)
            H1_repeat = H1_tensor.repeat(c_dim,1,1)
            # print(control_repeat.shape, H0_repeat.shape, H1_repeat.shape, dt.shape)
            U = torch.linalg.matrix_exp(-1.0j*dt_repeat*(H0_repeat + control_repeat*H1_repeat))
        else:
            U = torch.linalg.matrix_exp(-1.0j*dt*(H0_tensor + control*H1_tensor))

        # print("U", U)
        new_psi = U@psi

        reward = -costs.view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool, device=device)
        out = TensorDict(
            {
                "psi_real": torch.real(new_psi),
                "psi_imag": torch.imag(new_psi),
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
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

    PSI_0 = np.array([1, 0]).reshape((2,1))
    PSI_F = np.array([0, 1]).reshape((2,1))

    H0 = np.array([[1,  0],
                   [0, -1]])
    H1 = np.array([[0,  1],
                   [1,  0]])

    env = QuantumEnv(PSI_0=PSI_0, PSI_F=PSI_F, H0=H0, H1=H1)


    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)

    # print(env.reset())    

    # Test the environment
    check_env_specs(env)

    # Make the network
    torch.manual_seed(0)
    env.set_seed(0)

    class myNet(nn.Module):

        def __init__(self):
            super(myNet, self).__init__()
            self.net = nn.Sequential(
                        nn.Linear(4, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1),
                        nn.Tanh()
                        )

        def forward(self, x1, x2):
            # print(x1.shape, x2.shape)
            flattened = torch.squeeze(torch.hstack((x1, x2)))
            # print(flattened.shape)
            return self.net(flattened)

    net = myNet().to(device)
    policy = TensorDictModule(
        net,
        in_keys=["psi_real", "psi_imag"],
        out_keys=["action"],
    )
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Train
    batch_size = 1000
    rollout_len = 1000
    pbar = tqdm.tqdm(range(20_000 // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
    logs = defaultdict(list)

    for _ in pbar:
        init_td = env.reset(env.gen_params(batch_size=[batch_size]))
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
            plt.show()

    plot()