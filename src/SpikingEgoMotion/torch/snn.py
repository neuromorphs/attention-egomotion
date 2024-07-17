from typing import NamedTuple
import torch

from SpikingEgoMotion.torch.delta_lif import DeltaLIFCell, DeltaLIFParameters


class SNN(torch.nn.Module):
    """ SNN with delta-LIF """

    def __init__(
            self,
            kernel: torch.nn.Module,
            tau_mem: float = 1/10e-3,
            v_th: float = 1.,
            tau_mem_trainable: bool = False,
            v_th_trainable: bool = False):
        super().__init__()
        self.kernel = kernel
        self.tau_mem_trainable = tau_mem_trainable
        if tau_mem_trainable:
            self.tau_mem = torch.nn.Parameter(torch.tensor(tau_mem))
        else:
            self.tau_mem = torch.tensor(tau_mem)
        if v_th_trainable:
            self.v_th = torch.nn.Parameter(torch.tensor(v_th))
        else:
            self.v_th = torch.tensor(v_th)
        params = DeltaLIFParameters(self.tau_mem, 0., self.v_th)
        self.lif = DeltaLIFCell(p=params)

    def forward(self, input):
        T = input.shape[1]
        gs, vs, zs =  [], [], []
        if self.tau_mem_trainable:
            self.tau_mem.data = torch.max(
                torch.tensor(1/0.999), self.tau_mem.data)
        gs = self.kernel(input)
        state = self.lif.initial_state(None)
        for ts in range(T):
            z, state = self.lif(gs[:, ts], state)
            zs.append(z)
            vs.append(state.v)
        self.zs = torch.stack(zs).transpose(1, 0)
        self.vs = torch.stack(vs).transpose(1, 0)
        return self.zs
