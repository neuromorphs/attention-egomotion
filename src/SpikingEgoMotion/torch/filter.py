from typing import Optional, Tuple
import torch
import numpy as np



def gaussuian_filter(kernel_size, sigma=1, muu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2 + y**2)
    normal = 1 / np.sqrt(2 * np.pi * sigma**2)
    return dst, np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal


class TrainableEgoMotionRandomFilter(torch.nn.Module):

    def __init__(
            self, kernel_size, stride: int = 1, padding: int = 1,
            device = "cpu"):
        super().__init__()
        self.bias = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.conv = torch.nn.Conv2d(
            1, 1, kernel_size, stride=stride, padding=padding, bias=False,
            device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ys = []
        for ts in range(inputs.shape[1]):
            y = self.conv(inputs[:, ts])
            ys.append(y)
        return torch.stack(ys).transpose(1, 0)


class TrainableEgoMotionFilter(torch.nn.Module):

    def __init__(
            self, kernel_size, sigma1, sigma2, mu1, mu2, stride: int = 1,
            padding: int = 1, device = "cpu", normalize: bool = False,
            scale: Optional[Tuple[float, float]] = None):
        super().__init__()
        self.bias = None
        self.kernel_size = kernel_size
        self.sigma1 = torch.nn.Parameter(torch.tensor(sigma1))
        self.sigma2 = torch.nn.Parameter(torch.tensor(sigma2))
        self.mu1 = torch.nn.Parameter(torch.tensor(mu1))
        self.mu2 = torch.nn.Parameter(torch.tensor(mu2))
        self.stride = stride
        self.padding = padding
        self.device = device
        self.normalize = normalize
        if scale is not None:
            self.scale = torch.nn.Parameter(torch.tensor(scale))
        else:
            self.scale = torch.tensor([1., 1.])

        # Static plane
        x, y = np.meshgrid(np.linspace(-1, 1, self.kernel_size),
                           np.linspace(-1, 1, self.kernel_size))
        self.dst = torch.sqrt(
            torch.tensor(x).to(self.device)**2
            + torch.tensor(y).to(self.device)**2)

    def _kernel(self) -> torch.nn.Parameter:

        normal1 = 1 / torch.sqrt(2 * torch.pi * self.sigma1**2)
        normal2 = 1 / torch.sqrt(2 * torch.pi * self.sigma2**2)

        gauss1 = torch.exp(
            -((self.dst - self.mu1)**2 / (2.0 * self.sigma1**2))) * normal1
        gauss2 = torch.exp(
            -((self.dst - self.mu2)**2 / (2.0 * self.sigma2**2))) * normal2
        kernel = (self.scale[0] * gauss1 -
                  self.scale[1] * gauss2)[None, None, :].float()
        if self.normalize:
            kernel = kernel / kernel.sum()
        return kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ys = []
        n_kernel = self._kernel()
        for ts in range(inputs.shape[1]):
            y = torch.nn.functional.conv2d(
                inputs[:, ts], n_kernel, bias=self.bias, stride=self.stride,
                padding=self.padding)
            ys.append(y)
        return torch.stack(ys).transpose(1, 0)


class EgoMotionFilter(torch.nn.Module):

    def __init__(self, size, sigma1, sigma2, stride: int = 1, padding: int = 1):
        super().__init__()
        self.bias = None
        self.size = size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.stride = stride
        self.padding = padding
        self.kernel = self._kernel()

    def _kernel(self) -> torch.nn.Parameter:
        _, gauss1 = torch.tensor(
            gaussuian_filter(self.size, sigma=(self.sigma1)))
        _, gauss2 = torch.tensor(
            gaussuian_filter(self.size, sigma=(self.sigma2)))
        return torch.nn.Parameter((gauss1 - gauss2)[None, None, :].float())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            inputs, self.kernel, bias=self.bias, stride=self.stride,
            padding=self.padding)
    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kernel_size = 33
    filter = EgoMotionFilter(kernel_size, sigma1=0.05, sigma2=0.5)
    diff_gauss = filter.kernel.data

    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(diff_gauss)
    ax[1].plot(diff_gauss[:, 16])
    plt.savefig("./kernel_exmaple_2d.png")

    ax = plt.figure().add_subplot(projection='3d')
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    ax.plot_surface(
        x, y, diff_gauss, edgecolor='royalblue', lw=0.5, rstride=1, cstride=1,
        alpha=0.3)

    plt.savefig("./kernel_exmaple_3d.png")
