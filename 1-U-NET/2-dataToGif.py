import torch
import matplotlib.pyplot as plt
import os

def gifCreator(file):
    dataset = torch.load(f'../dataset/egomotion/{file}.pt')

    interval = dataset['source'].shape[0]
    for t in range(interval):
        print(f'frame {t+1}/{interval}')
        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        plt.cla()
        plt.title('Source')
        plt.imshow(dataset['source'][t], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.cla()
        plt.title(f'{file}\nTarget')
        plt.imshow(dataset['target'][t], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.cla()
        plt.title('No egomotion')
        plt.imshow(dataset['targetStack'][t], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f'../image/{file}{t}.png')
        plt.close()

    os.system(f'ffmpeg -framerate 10 -y -i ../image/{file}%d.png ../image/{file}.gif')
    os.system(f'rm -r ../image/*.png')
    os.system(f'rm -r ../dataset/egomotion/{file}.pt')


if __name__ == '__main__':
    gifCreator('sphereInference')
    gifCreator('sphereInferenceSpike')
