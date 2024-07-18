import torch
import matplotlib.pyplot as plt

def datasetCreation(name, vx, vy):
    frame, witdth, height = 1000, 400, 400

    source = torch.zeros((frame, witdth, height))
    target = torch.zeros((frame, witdth, height))

    background = torch.randint(2, (witdth, height))
    backgroundXShift, backgroundYShift = 10, 0


    object = torch.zeros_like(background)
    xCenter, yCenter = 50, 50
    radius = 20
    angles = torch.linspace(0, torch.pi/2, 100)
    if 'square' in name:
            object[xCenter-radius:xCenter+radius, yCenter-radius:yCenter+radius] = 1
    else:
        x, y = (radius * torch.cos(angles)).type(torch.int), (radius * 1.5 * torch.sin(angles)).type(torch.int)
        for a, b in zip(x, y):
            object[xCenter:xCenter+a, yCenter:yCenter+b] = 1
            object[xCenter-a:xCenter, yCenter:yCenter+b] = 1
            object[xCenter:xCenter+a, yCenter-b:yCenter] = 1
            object[xCenter-a:xCenter, yCenter-b:yCenter] = 1
    objectXShift, objectYShift = vx, vy


    # plt.figure()

    for i in range(frame):
        scene = torch.where(background+object > 1, 1, background+object)

        # plt.cla()
        # plt.imshow(scene, cmap='gray')
        # plt.pause(0.1)

        background = torch.roll(background, backgroundXShift, dims=1)
        background = torch.roll(background, backgroundYShift, dims=0)

        object = torch.roll(object, objectXShift, dims=1)
        object = torch.roll(object, objectYShift, dims=0)

        source[i, :, :] = scene
        target[i, :, :] = object

    dataset = {'source': source, 'target': target}
    torch.save(dataset, f'../dataset/egomotion/{name}.pt')

    # plt.close()


if __name__ == '__main__':
    datasetCreation('sphere', 17, 27)
    datasetCreation('ellipsoid', 17, -27)
    datasetCreation('square', -3, 27)
