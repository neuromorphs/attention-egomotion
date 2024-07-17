import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class EgoMotionDataset(torch.utils.data.Dataset):

    def __init__(
            self, size, width, height, velocity: list, obj_size: int = 10,
            n_objects: int = 5, noise_level: int = 0.1, shift: int = 10,
            period: int = 500, period_sim = 10000):
        self.size = size
        self.obj_size = obj_size
        self.velocity = velocity
        self.width = width
        self.height = height
        self.n_objects = n_objects
        self.noise_level = noise_level
        self.shift = shift
        self.period = period
        self.period_sim = period_sim

    def __len__(self):
        return self.size

    def _generate_sample(self):

        time_slices = []
        coherent_noise = np.random.uniform(
            size=(self.height + 2*self.obj_size, self.width + 2*self.obj_size)
            ) < self.noise_level
        for _ in range(int(self.period_sim / self.period)):
            coherent_noise = np.roll(coherent_noise, shift=self.shift, axis=1)
            time_slices.append(torch.tensor(coherent_noise))
        noise = torch.stack(time_slices)

        objects = torch.zeros_like(noise)

        for n in range(self.n_objects):
            direction = np.random.uniform(size=2) * (
                np.random.randint(0, 2, size=2)*2-1)
            direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
            offset = [
                np.random.randint(self.obj_size, high=self.width),
                np.random.randint(self.obj_size, high=self.height)]
            velocity = self.velocity[0] + np.random.rand() / (
                self.velocity[1] - self.velocity[0])

            Y, X = np.ogrid[:self.obj_size, :self.obj_size]
            center = (int(self.obj_size / 2), int(self.obj_size / 2))
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = torch.tensor(dist_from_center <= self.obj_size // 2)

            for i, n in enumerate(objects):
                x = direction[0] * i * velocity + offset[0]
                x = int(direction[0] * i * velocity + offset[0])
                x = max(0, min(x, self.width + self.obj_size))
                y = int(direction[1] * i * velocity + offset[1])
                y = max(0, min(y, self.height + self.obj_size))
                n[y: y + self.obj_size, x: x + self.obj_size][mask] = 1

        sample = noise + objects
        sample = sample[:, self.obj_size:-self.obj_size,
                        self.obj_size:-self.obj_size]
        target = objects[:, self.obj_size:-self.obj_size,
                         self.obj_size:-self.obj_size]
        return sample.unsqueeze(1).float(), target.unsqueeze(1).float()

    def __getitem__(self, index: int):
        sample = self._generate_sample()
        return sample


if __name__ == "__main__":
    shift = 10 # pixels
    period = 500 # millisecods
    v = shift / period  # px/s ; please put the speed in seconds
    period_sim = 10000 # millisecods
    width = 304
    height = 240

    dataset = EgoMotionDataset(
        1, width, height, velocity=(
            (period_sim / period) / np.array(
            [period_sim / period * 1.2, period_sim / period])))
    sample, _ = dataset[0]
    print(sample.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(sample[0, 0], cmap='gray')

    def animate(step):
        print(step)
        im.set_data(sample[step, 0])
        return [im]

    ani = FuncAnimation(
        fig, animate, interval=period, blit=True, repeat=True,
        frames=sample.shape[0])
    ani.save("sample.gif", dpi=300, writer=PillowWriter(fps=100))