import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from SpikingEgoMotion.torch.filter import (
    TrainableEgoMotionFilter, TrainableEgoMotionRandomFilter)
from SpikingEgoMotion.torch.datasets import EgoMotionDataset
from SpikingEgoMotion.torch.snn import SNN


# Data
base_path = Path("./data/spiking_random_kernel")
data_path = Path("data.csv")
model_path = Path("model.pt")
base_path.mkdir(exist_ok=True, parents=True)

# SNN
tau_mem = 1 / 5e-3
tau_mem_trainable = True
v_th = 0.5
v_th_trainable = True

# Pattern
shift = 10 # pixels
period = 500 # millisecods
v = shift / period  # px/s ; please put the speed in seconds
period_sim = 10000 # millisecods
width = 304
height = 240
noise_level = 0.1

# Filter
kernel_size = 33
sigma1 = 0.3
sigma2 = 0.3
normalize = False
kernel_name = "random"
scale = [1., 1.]

# Training
gamma = 0.9
step = 10
epochs = 10
batch_size = 100
lr = 1e-3
lr_tau_mem = 5e-1
lr_v_th = 5e-4
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float

# Fix random seed
np.random.seed(0)
torch.manual_seed(0)


def van_rossum_pixels(output, target, decay: float = 0.5):
    target_conv = [target[:, 0]]
    output_conv = [output[:, 0]]
    target_data = target[:, 0] 
    output_data = output[:, 0] 
    for ts in range(1, output.shape[1]):
        output_data = decay * output_data + output[:, ts]
        target_data = decay * target_data + target[:, ts]
        output_conv.append(output_data)
        target_conv.append(target_data)
    target_conv = torch.stack(target_conv).transpose(1, 0)
    output_conv = torch.stack(output_conv).transpose(1, 0)
    return target_conv - output_conv

def van_rossum(output, target, decay: float = 0.5):
    return torch.sqrt(
        torch.sum((van_rossum_pixels(output, target)**2)) * decay)
loss_fn = van_rossum


def do_epoch(model, data_loader, optimizer, scheduler, training: bool):
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

        scheduler.step()


    pbar.close()

    loss = torch.stack(losses) / samples

    return loss


def get_model():
    if kernel_name == "random":
        kernel = TrainableEgoMotionRandomFilter(
            kernel_size, device=device, padding=int(kernel_size//2))
    if kernel_name == "gauss":
        kernel = TrainableEgoMotionFilter(
            kernel_size, sigma1=sigma1, sigma2=sigma2, mu1=0., mu2=0.,
            device=device, padding=int(kernel_size//2), normalize=normalize,
            scale=scale)
    return SNN(kernel, tau_mem, v_th, tau_mem_trainable, v_th_trainable)


def main():
    # Data
    velocity = ((period_sim / period) / np.array(
            [period_sim / period * 1.2, period_sim / period]))
    train_set = EgoMotionDataset(
        10000, width, height, velocity=velocity, noise_level=noise_level)
    val_set = EgoMotionDataset(
        2000, width, height, velocity=velocity, noise_level=noise_level)

    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=1)

    # Model
    model = get_model()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": model.tau_mem, "lr": lr_tau_mem},
        {"params": model.v_th, "lr": lr_v_th},
        {"params": model.kernel.parameters(), "lr": lr}], betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step, gamma=gamma)

    # Train and test
    data = torch.zeros((epochs, len(train_loader), batch_size))

    pbar = tqdm(total=epochs, unit="epoch", position=0)
    for epoch in range(epochs):
        # Train and evaluate
        train_loss = do_epoch(model, train_loader, optimizer, scheduler, True)
        # val_loss = do_epoch(model, val_loader, optimizer, scheduler, False)

        pbar.set_postfix(loss=f"{train_loss.sum():.4f}")
        pbar.update()

        # Keep
        data[0, epoch] = train_loss
        # data[1, epoch] = val_loss

        with open(base_path / data_path, "wb") as file:
            np.save(file, data.numpy())

        # Save model
        torch.save(model.to("cpu").state_dict(), base_path / model_path)
        model.to(device)


if __name__ == "__main__":
    main()