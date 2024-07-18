import torch
from datasetClass import DatasetClassEgomotion
from models import Model
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
# ##### Model definition ##### #
################################
neuralNetwork = Model().to(device)
optimizer = torch.optim.AdamW(neuralNetwork.parameters(), lr=3e-4)
lossFunction = torch.nn.BCEWithLogitsLoss()


#############################
# ##### Training Loop ##### #
#############################
def trainingLoop():
    for epoch in range(10):
        neuralNetwork.train()
        lossRun = 0
        for i, (source, target) in enumerate(datasetDL):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            loss = lossFunction(neuralNetwork(source), target)
            loss.backward()
            optimizer.step()

            lossRun += loss.item()

        lossRun /= (i+1)

        print(f'Epoch {epoch+1:03d} | Loss Train: {lossRun:.3f}')

        torch.save(neuralNetwork.state_dict(), '../model/egomotion.pt')


##############################
# ##### Inference Loop ##### #
##############################
def inferenceLoop(datasetName, interval, live=True):
    ##############################
    # ##### Plot inference ##### #
    ##############################
    dataset['source'] = dataset['source'][0:interval]
    dataset['target'] = dataset['target'][0:interval]

    neuralNetwork.eval()
    datasetClass = DatasetClassEgomotion(dataset)
    datasetDL = torch.utils.data.DataLoader(datasetClass, batch_size=1)
    targetStack = []
    for i, (source, target) in enumerate(datasetDL):
        print(f'frame {i + 1}/{interval}')
        source, target = source.to(device), target.to(device)
        targetPred = neuralNetwork(source)

        targetPred = targetPred.squeeze(1)
        targetPred = targetPred.detach().cpu()
        targetPred[targetPred > 0] = 1
        targetPred[targetPred < 0] = 0
        targetStack.append(targetPred)

    targetStack = torch.concatenate(targetStack, dim=0)

    dataset['targetStack'] = targetStack

    if live is False:
        torch.save(dataset, f'../dataset/egomotion/{datasetName}Inference.pt')
    else:
        plt.figure(figsize=(9, 3))
        for t in range(interval):
            plt.subplot(1, 3, 1)
            plt.cla()
            plt.title('Source')
            plt.imshow(dataset['source'][t], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 3, 2)
            plt.cla()
            plt.title('Target')
            plt.imshow(dataset['target'][t], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1, 3, 3)
            plt.cla()
            plt.title('No egomotion')
            plt.imshow(dataset['targetStack'][t], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.pause(0.1)


if __name__ == '__main__':
    ##################################
    # ##### Dataset definition ##### #
    ##################################
    datasetName = 'sphere'
    dataset = torch.load(f'../dataset/egomotion/{datasetName}.pt')

    datasetClass = DatasetClassEgomotion(dataset)
    batchSize = 5
    datasetDL = torch.utils.data.DataLoader(datasetClass, batch_size=batchSize, shuffle=True)

    # trainingLoop()
    parameters = torch.load('../model/egomotion.pt', map_location=device)
    neuralNetwork.load_state_dict(parameters)

    inferenceLoop(datasetName, 29, live=False)
