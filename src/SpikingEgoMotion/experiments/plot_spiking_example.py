from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from SpikingEgoMotion.torch.filter import (
    TrainableEgoMotionFilter, TrainableEgoMotionRandomFilter)
from matplotlib.animation import FuncAnimation, PillowWriter
from SpikingEgoMotion.torch.datasets import EgoMotionDataset
from SpikingEgoMotion.torch.snn import SNN
from SpikingEgoMotion.experiments.spiking_train_filter import van_rossum_pixels

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
path = Path("./data/spiking_random_kernel")


def plot(path):

    period = 500 # millisecods
    period_sim = 10000 # millisecods
    width = 304
    height = 240

    kernel_size = 33
    sigma1 = 0.05
    sigma2 = 0.5
    scale = [1., 1.]

    kernel = "spiking_random"
    normalize = False
    tau_mem = 1 / 5e-3
    v_th = 0.005
    tau_mem_trainable = True
    v_th_trainable = True

    model_dict = torch.load(path / "model.pt")
    if kernel == "spiking_gauss":
        kernel = TrainableEgoMotionFilter(
            kernel_size, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
            device=device, padding=int(kernel_size//2), normalize=normalize,
            scale=scale)
        model = SNN(kernel, tau_mem, v_th, tau_mem_trainable, v_th_trainable)
        model.load_state_dict(model_dict)
    if kernel == "spiking_random":
        kernel = TrainableEgoMotionRandomFilter(
            kernel_size, device=device, padding=int(kernel_size//2))
        model = SNN(kernel, tau_mem, v_th, tau_mem_trainable, v_th_trainable)
        model.load_state_dict(model_dict)

    dataset = EgoMotionDataset(
        1000, width, height, velocity=(
            (period_sim / period) / np.array(
            [period_sim / period * 1.2, period_sim / period])), noise_level=0.1)
    sample, target = dataset[0]

    y = model(sample.float().unsqueeze(0).to(device))
    v = model.vs.cpu().detach().numpy()[0]
    van_rossum = van_rossum_pixels(y, target.unsqueeze(0), decay=0.5)
    y = y.cpu().detach().numpy()[0]
    van_rossum = van_rossum.cpu().detach().numpy()[0]

    b = int(kernel_size//2)
    sample = sample[:, :, b:-b, b:-b]
    target = target[:, :, b:-b, b:-b]
    y = y[:, :, b:-b, b:-b]
    v = v[:, :, b:-b, b:-b]
    van_rossum = van_rossum[:, :, b:-b, b:-b]

    fig, ax = plt.subplots(ncols=5, figsize=(16, 4))
    im1 = ax[0].imshow(sample[0, 0], vmin=0, vmax=1, cmap='gray')
    im2 = ax[1].imshow(target[0, 0], vmin=0, vmax=1, cmap='gray')
    im3 = ax[2].imshow(
        v[0, 0], vmin=-model.v_th.data.item(), vmax=model.v_th.data.item(),
        cmap="bwr")
    im4 = ax[3].imshow(y[0, 0], vmin=0, vmax=1, cmap='gray')
    im5 = ax[4].imshow(van_rossum[0, 0], vmin=-1, vmax=1, cmap="bwr")

    ax[0].set_title("Input")
    ax[1].set_title("Target")
    ax[2].set_title("Membrane")
    ax[3].set_title("Spikes")
    ax[4].set_title("Van Rossum Distance")

    def animate(step):
        im1.set_data(sample[step, 0])
        im2.set_data(target[step, 0])
        im3.set_data(v[step, 0])
        im4.set_data(y[step, 0])
        im5.set_data(van_rossum[step, 0])
        return [im1, im2, im3, im4, im5]

    ani = FuncAnimation(
        fig, animate, interval=period, blit=True, repeat=True,
        frames=sample.shape[0])
    ani.save(path / "sample.gif", dpi=300, writer=PillowWriter(fps=100))


if __name__ == "__main__":
    plot(path)