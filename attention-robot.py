import sinabs
import sinabs.layers as sl
import torch
import torch.nn as nn
import numpy as np
import tonic
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

matplotlib.use('TkAgg')

def create_fltrs(fltr_resize_perc, angle_shift):
    angles = range(0, 360, angle_shift)
    filters = []
    for i in angles:
        filter = np.load(f"VMfilters/{i}_grad.npy")
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    # tensor with 8 orientation VM filter
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters

def net_def(filters):
    # define our single layer network and load the filters
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[0], filters.shape[1], bias=False),
        sl.IAF()
    )
    net[0].weight.data = filters.unsqueeze(1)
    return net


def run(net, frames, max_x, max_y, num_pyr):
    salmap=torch.empty((1, num_pyr, max_y, max_x), dtype=torch.int64)
    #scales pyramid
    for pyr in range(1, num_pyr+1):
        print(f"pyramid scale {pyr}")
        res = (int(max_y/pyr), int(max_x/pyr))
        #resize input for the pyramid
        frm_rsz = torchvision.transforms.Resize((res[0], res[1]))(frames)
        # now we feed the data to our network! Because my computer has little memory, I only feed 10 specific time steps
        with torch.no_grad():
            output = net(frm_rsz.float())
        output.shape
        #sum over different rotations
        output_rotations = torch.sum(output, dim=0, keepdim=True)
        output_rotresz = torchvision.transforms.Resize((max_y, max_x))(output_rotations)
        salmap[0,(pyr-1)] = torchvision.transforms.Resize((max_y, max_x))(output_rotresz[0])
    salmap = torch.sum(salmap, dim=1, keepdim=True)
    return salmap, filter.shape

if __name__ == '__main__':
    # load all the filters and stack them to a 3d array of (filter number, width, height)
    angle_shift = 45
    fltr_resize_perc = [1.8, 1.9]
    filters = create_fltrs(fltr_resize_perc, angle_shift)
    net = net_def(filters)
    img = Image.open("lovelycat.jpeg").convert('L')
    convert_tensor = transforms.ToTensor()
    frames = convert_tensor(img)
    max_y, max_x = frames[0].size()
    num_pyr = 4
    run(net, frames, max_x, max_y, num_pyr)