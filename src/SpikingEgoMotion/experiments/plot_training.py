from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot(path):

    data = np.load(path / "data.csv")

    # First layer
    fig, axs = plt.subplots()
    axs.plot(data[0], label="train")
    axs.plot(data[1], label="test")
    axs.legend()
    plt.savefig(path / "training.png", dpi=200)

if __name__ == "__main__":
    path = Path("./data/gauss_kernel")