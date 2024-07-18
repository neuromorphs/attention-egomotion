import numpy as np
import torch
import matplotlib.pyplot as plt

labelled_events = np.load('../dataset/labelled_events.npy')


time = labelled_events['t']*1e-6
bins = np.linspace(0, 30, 30)
intervals = np.searchsorted(bins, time)-1

background = torch.zeros((29, 400, 400))
objects = torch.zeros((29, 400, 400))

for i, t in enumerate(intervals):
    _, x, y, _, label = labelled_events[i]
    if 400 <= x < 800 and 400 <= y < 800:
        if label < 0:
            objects[t, x-400, y-400] = 1
        else:
            background[t, x-400, y-400] = 1

plt.figure()

for i in range(29):
    scene = torch.where(background[i]+objects[i] > 1, 1, background[i]+objects[i])

    plt.cla()
    plt.imshow(scene, cmap='gray')
    plt.pause(0.1)

dataset = {'source': background, 'target': objects}
torch.save(dataset, f'../dataset/egomotion/realData.pt')

plt.close()
