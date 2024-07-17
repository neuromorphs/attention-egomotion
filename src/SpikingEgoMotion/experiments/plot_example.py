from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from SpikingEgoMotion.torch.filter import (
    TrainableEgoMotionFilter, TrainableEgoMotionRandomFilter)
from matplotlib.animation import FuncAnimation, PillowWriter
from SpikingEgoMotion.torch.datasets import EgoMotionDataset

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
path = Path("./data/spiking_gauss_kernel")


def plot(path):

    period = 500 # millisecods
    period_sim = 10000 # millisecods
    width = 304
    height = 240

    kernel_size = 33
    sigma1 = 0.05
    sigma2 = 0.5
    kernel = "gauss"

    model_dict = torch.load(path / "model.pt")
    if kernel == "gauss":
        model = TrainableEgoMotionFilter(
            33, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
            device=device, padding=int(kernel_size//2))
        model.load_state_dict(model_dict)
    if kernel == "random":
        model = TrainableEgoMotionRandomFilter(
            33, device=device, padding=int(kernel_size//2))
        model.load_state_dict(model_dict)

    dataset = EgoMotionDataset(
        1000, width, height, velocity=(
            (period_sim / period) / np.array(
            [period_sim / period * 1.2, period_sim / period])))
    sample, target = dataset[0]

    y = model(sample.float().unsqueeze(0).to(device))
    y = y.cpu().detach().numpy()[0]
    y = (y - y.mean()) / y.std()
    y = (y - y.min()) / (y.max() - y.min())

    # Binary thresholding
    y_th = y.copy()
    y_th[y < 0.5] = 0
    y_th[y >= 0.5] = 1

    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    im1 = ax[0].imshow(sample[0, 0], vmin=0, vmax=1, cmap='gray')
    im2 = ax[1].imshow(target[0, 0], vmin=0, vmax=1, cmap='gray')
    im3 = ax[2].imshow(y[0, 0], vmin=0, vmax=1, cmap='gray')
    im4 = ax[3].imshow(y_th[0, 0], vmin=0, vmax=1, cmap='gray')

    ax[0].set_title("Orginal")
    ax[1].set_title("Target")
    ax[2].set_title("Filtered")
    ax[3].set_title("Thresholded")

    def animate(step):
        im1.set_data(sample[step, 0])
        im2.set_data(target[step, 0])
        im3.set_data(y[step, 0])
        im4.set_data(y_th[step, 0])
        return [im1, im2, im3, im4]

    ani = FuncAnimation(
        fig, animate, interval=period, blit=True, repeat=True,
        frames=sample.shape[0])
    ani.save(path / "sample.gif", dpi=300, writer=PillowWriter(fps=100))


if __name__ == "__main__":
    plot(path)