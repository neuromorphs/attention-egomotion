from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from SpikingEgoMotion.torch.filter import (
    TrainableEgoMotionFilter, TrainableEgoMotionRandomFilter)
from SpikingEgoMotion.torch.snn import SNN


device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
path = Path("./data/spiking_random_kernel")


def plot(path):

    kernel_size = 33
    sigma1 = 0.05
    sigma2 = 0.5
    kernel = "spiking_random"
    normalize = False
    scale = [1., 1.]

    tau_mem = 1 / 3e-6
    v_th = 1.
    tau_mem_trainable = True
    v_th_trainable = True

    model_dict = torch.load(path / "model.pt")
    if kernel == "gauss":
        model = TrainableEgoMotionFilter(
            33, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
            device=torch.device("cpu"), padding=int(kernel_size//2))
        model.load_state_dict(model_dict)
        print(model.sigma1)
        print(model.sigma2)
        print(model.mu1)
        print(model.mu2)
        kernel = model._kernel()
    if kernel == "random":
        model = TrainableEgoMotionRandomFilter(
            33, device=torch.device("cpu"), padding=int(kernel_size//2))
        model.load_state_dict(model_dict)
        kernel = model.conv.weight
    if kernel == "spiking_gauss":
        kernel = TrainableEgoMotionFilter(
            kernel_size, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
            device=device, padding=int(kernel_size//2), normalize=normalize,
            scale=scale)
        model = SNN(kernel, tau_mem, v_th, tau_mem_trainable, v_th_trainable)
        model.load_state_dict(model_dict)
        kernel = model.kernel._kernel()
    if kernel == "spiking_random":
        kernel = TrainableEgoMotionRandomFilter(
            kernel_size, device=device, padding=int(kernel_size//2))
        model = SNN(kernel, tau_mem, v_th, tau_mem_trainable, v_th_trainable)
        model.load_state_dict(model_dict)
        kernel = model.kernel.conv.weight

    # First layer
    fig, axs = plt.subplots()
    v_min = kernel.min()
    v_max = kernel.max()
    axs.imshow(kernel.detach().numpy()[0, 0], vmin=v_min, vmax=v_max)
    plt.savefig(path / "kernel.png", dpi=200)

    ax = plt.figure().add_subplot(projection='3d')
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    ax.plot_surface(
        x, y, kernel.detach().numpy()[0, 0], edgecolor='royalblue', lw=0.5, 
        rstride=1, cstride=1,
        alpha=0.3)
    plt.savefig(path / "kernel_3d.png", dpi=200)

if __name__ == "__main__":
    plot(path)