import torchvision
import numpy as np
import matplotlib.pyplot as plt


def plotSourceTarget(source, target, model, device):
    source, target = source.to(device), target.to(device)
    targetPred = model(source)

    source = torchvision.utils.make_grid(source).detach().cpu().numpy()
    target = torchvision.utils.make_grid(target).detach().cpu().numpy()
    targetPred = torchvision.utils.make_grid(targetPred).detach().cpu().numpy()

    source, target = np.transpose(source, (1, 2, 0)), np.transpose(target, (1, 2, 0))
    targetPred = np.transpose(targetPred, (1, 2, 0))
    targetPred[targetPred > 0] = 1
    targetPred[targetPred < 0] = 0

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(source)
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(target)
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(targetPred)
    plt.axis('off')


def plotSourceTargetSpike(source, target, model, device):
    source, target = source.to(device), target.to(device)
    spikePred, targetPred = model(source)

    source = torchvision.utils.make_grid(source).detach().cpu().numpy()
    target = torchvision.utils.make_grid(target).detach().cpu().numpy()
    targetPred = torchvision.utils.make_grid(targetPred[-1]).detach().cpu().numpy()

    source, target = np.transpose(source, (1, 2, 0)), np.transpose(target, (1, 2, 0))
    targetPred = np.transpose(targetPred, (1, 2, 0))
    targetPred[targetPred > 0] = 1
    targetPred[targetPred < 0] = 0

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(source)
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(target)
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(targetPred)
    plt.axis('off')