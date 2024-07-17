from typing import NamedTuple, Tuple
import torch
import norse.torch.functional as F
from norse.torch.module.snn import SNNCell


class DeltaLIFFeedForwardState(NamedTuple):
    v: torch.Tensor


class DeltaLIFParameters(NamedTuple):
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(50.0)


def delta_lif_step(
        input_spikes: torch.Tensor,
        state: DeltaLIFFeedForwardState,
        p: DeltaLIFParameters,
        dt: float = 0.001) -> Tuple[torch.Tensor, DeltaLIFFeedForwardState]:
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + input_spikes)
    v_decayed = state.v + dv
    # compute new spikes
    z_new = F.threshold.threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    return z_new, DeltaLIFFeedForwardState(v=v_new)


class DeltaLIFCell(SNNCell):

    def __init__(self, p: DeltaLIFParameters = DeltaLIFParameters(), **kwargs):
        super().__init__(
            activation=delta_lif_step,
            activation_sparse=None,
            state_fallback=self.initial_state,
            p=p,
            **kwargs)

    def initial_state(self, input: torch.Tensor) \
            -> DeltaLIFFeedForwardState:
        state = DeltaLIFFeedForwardState(
            v=torch.as_tensor(self.p.v_leak))
        state.v.requires_grad = True
        return state