import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from SpikingEgoMotion.torch.filter import (
    TrainableEgoMotionFilter, TrainableEgoMotionRandomFilter)
from SpikingEgoMotion.torch.datasets import EgoMotionDataset


# Data
base_path = Path("./data/gauss_kernel_normalized")
data_path = Path("data.csv")
model_path = Path("model.pt")
base_path.mkdir(exist_ok=True, parents=True)

# Pattern
shift = 10 # pixels
period = 500 # millisecods
v = shift / period  # px/s ; please put the speed in seconds
period_sim = 10000 # millisecods
width = 304
height = 240

# Filter
kernel_size = 33
sigma1 = 0.05
sigma2 = 0.5
normalize = True
kernel = "gauss"

# Training
epochs = 100
batch_size = 100
lr = 1e-4
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float
loss_fn = torch.nn.MSELoss()

# Fix random seed
np.random.seed(0)
torch.manual_seed(0)


def do_epoch(model, data_loader, optimizer, training: bool):
    model.train(training)

    # Minibatch training loop
    losses = []
    samples = 0
    pbar = tqdm(total=len(data_loader), unit="batch", position=1, leave=True)
    for data, target in data_loader:

        data = data.to(device).to(dtype)
        target = target.to(device).to(dtype)

        if training:
            optimizer.zero_grad()

        # forward pass (one polarity)
        y = model(data)

        #  loss
        loss = loss_fn(y, target)

        # Gradient calculation + weight update
        if training:
            loss.backward()
            optimizer.step()

        # Store loss history for future plotting
        losses.append(loss.detach() * data.shape[0])

        # count samples
        samples += data.shape[0]

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update()


    pbar.close()

    loss = torch.stack(losses).sum() / samples

    return loss


def get_model():
    if kernel == "random":
        return TrainableEgoMotionRandomFilter(
            kernel_size, device=device, padding=int(kernel_size//2))
    return TrainableEgoMotionFilter(
        kernel_size, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
        device=device, padding=int(kernel_size//2), normalize=normalize)


def main():
    # Data
    velocity = ((period_sim / period) / np.array(
            [period_sim / period * 1.2, period_sim / period]))
    train_set = EgoMotionDataset(10000, width, height, velocity=velocity)
    val_set = EgoMotionDataset(2000, width, height, velocity=velocity)

    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, batch_size=batch_size, num_workers=1)

    # Model
    model = get_model()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # Train and test
    data = torch.zeros((2, epochs))

    pbar = tqdm(total=epochs, unit="epoch", position=0)
    for epoch in range(epochs):
        # Train and evaluate
        train_loss = do_epoch(model, train_loader, optimizer, True)
        val_loss = do_epoch(model, val_loader, optimizer, False)

        pbar.set_postfix(loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        pbar.update()

        # Keep
        data[0, epoch] = train_loss
        data[1, epoch] = val_loss

        with open(base_path / data_path, "wb") as file:
            np.save(file, data.numpy())

        # Save model
        torch.save(model.to("cpu").state_dict(), base_path / model_path)
        model.to(device)


if __name__ == "__main__":
    main()