import torch
import snntorch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1_1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.activ1_1 = torch.nn.ReLU(True)
        self.layer1_2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.activ1_2 = torch.nn.ReLU(True)
        self.pool1    = torch.nn.MaxPool2d(2, 2)

        self.layer2_1 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.activ2_1 = torch.nn.ReLU(True)
        self.layer2_2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.activ2_2 = torch.nn.ReLU(True)
        self.pool2    = torch.nn.MaxPool2d(2, 2)

        self.layer3_1 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.activ3_1 = torch.nn.ReLU(True)
        self.layer3_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.activ3_2 = torch.nn.ReLU(True)
        self.pool3    = torch.nn.MaxPool2d(2, 2)

        self.layer4_1 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.activ4_1 = torch.nn.ReLU(True)
        self.layer4_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.activ4_2 = torch.nn.ReLU(True)
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        self.layer5_1 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.activ5_1 = torch.nn.ReLU(True)
        self.layer5_2 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.activ5_2 = torch.nn.ReLU(True)
        self.unpool5 = torch.nn.ConvTranspose2d(256, 128, 2, 2)

        self.layer6_1 = torch.nn.Conv2d(256, 128, 3, 1, 1)
        self.activ6_1 = torch.nn.ReLU(True)
        self.layer6_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.activ6_2 = torch.nn.ReLU(True)
        self.unpool6 = torch.nn.ConvTranspose2d(128, 64, 2, 2)

        self.layer7_1 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self.activ7_1 = torch.nn.ReLU(True)
        self.layer7_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.activ7_2 = torch.nn.ReLU(True)
        self.unpool7 = torch.nn.ConvTranspose2d(64, 32, 2, 2)

        self.layer8_1 = torch.nn.Conv2d(64, 32, 3, 1, 1)
        self.activ8_1 = torch.nn.ReLU(True)
        self.layer8_2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.activ8_2 = torch.nn.ReLU(True)
        self.unpool8 = torch.nn.ConvTranspose2d(32, 16, 2, 2)

        self.layer9_1 = torch.nn.Conv2d(32, 16, 3, 1, 1)
        self.activ9_1 = torch.nn.ReLU(True)
        self.layer9_2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.activ9_2 = torch.nn.ReLU(True)
        self.layer9_3 = torch.nn.Conv2d(16, 1, 1)

    def forward(self, batchInput):
        batch = self.layer1_1(batchInput)
        batch = self.activ1_1(batch)
        batch = self.layer1_2(batch)
        batchRes1 = self.activ1_2(batch)
        batch = self.pool1(batchRes1)

        batch = self.layer2_1(batch)
        batch = self.activ2_1(batch)
        batch = self.layer2_2(batch)
        batchRes2 = self.activ2_2(batch)
        batch = self.pool2(batchRes2)

        batch = self.layer3_1(batch)
        batch = self.activ3_1(batch)
        batch = self.layer3_2(batch)
        batchRes3 = self.activ3_2(batch)
        batch = self.pool3(batchRes3)

        batch = self.layer4_1(batch)
        batch = self.activ4_1(batch)
        batch = self.layer4_2(batch)
        batchRes4 = self.activ4_2(batch)
        batch = self.pool4(batchRes4)

        batch = self.layer5_1(batch)
        batch = self.activ5_1(batch)
        batch = self.layer5_2(batch)
        batch = self.activ5_2(batch)
        batch = self.unpool5(batch)

        batch = self.residualConnection(batchRes4, batch)
        batch = self.layer6_1(batch)
        batch = self.activ6_1(batch)
        batch = self.layer6_2(batch)
        batch = self.activ6_2(batch)
        batch = self.unpool6(batch)

        batch = self.residualConnection(batchRes3, batch)
        batch = self.layer7_1(batch)
        batch = self.activ7_1(batch)
        batch = self.layer7_2(batch)
        batch = self.activ7_2(batch)
        batch = self.unpool7(batch)

        batch = self.residualConnection(batchRes2, batch)
        batch = self.layer8_1(batch)
        batch = self.activ8_1(batch)
        batch = self.layer8_2(batch)
        batch = self.activ8_2(batch)
        batch = self.unpool8(batch)

        batch = self.residualConnection(batchRes1, batch)
        batch = self.layer9_1(batch)
        batch = self.activ9_1(batch)
        batch = self.layer9_2(batch)
        batch = self.activ9_2(batch)
        batchOutput = self.layer9_3(batch)

        return batchOutput

    def residualConnection(self, tensorSource, tensorTarget):
        return torch.cat([tensorSource, tensorTarget], 1)


class ModelSpike(torch.nn.Module):
    def __init__(self, numSteps):
        super().__init__()

        beta = 0.95

        self.layer1_1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.activ1_1 = snntorch.Leaky(beta=beta)
        self.layer1_2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.activ1_2 = snntorch.Leaky(beta=beta)
        self.pool1    = torch.nn.MaxPool2d(2, 2)

        self.layer2_1 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.activ2_1 = snntorch.Leaky(beta=beta)
        self.layer2_2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.activ2_2 = snntorch.Leaky(beta=beta)
        self.pool2    = torch.nn.MaxPool2d(2, 2)

        self.layer3_1 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.activ3_1 = snntorch.Leaky(beta=beta)
        self.layer3_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.activ3_2 = snntorch.Leaky(beta=beta)
        self.pool3    = torch.nn.MaxPool2d(2, 2)

        self.layer4_1 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.activ4_1 = snntorch.Leaky(beta=beta)
        self.layer4_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.activ4_2 = snntorch.Leaky(beta=beta)
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        self.layer5_1 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.activ5_1 = snntorch.Leaky(beta=beta)
        self.layer5_2 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.activ5_2 = snntorch.Leaky(beta=beta)
        self.unpool5 = torch.nn.ConvTranspose2d(256, 128, 2, 2)

        self.layer6_1 = torch.nn.Conv2d(256, 128, 3, 1, 1)
        self.activ6_1 = snntorch.Leaky(beta=beta)
        self.layer6_2 = torch.nn.Conv2d(128, 128, 3, 1, 1)
        self.activ6_2 = snntorch.Leaky(beta=beta)
        self.unpool6 = torch.nn.ConvTranspose2d(128, 64, 2, 2)

        self.layer7_1 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self.activ7_1 = snntorch.Leaky(beta=beta)
        self.layer7_2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.activ7_2 = snntorch.Leaky(beta=beta)
        self.unpool7 = torch.nn.ConvTranspose2d(64, 32, 2, 2)

        self.layer8_1 = torch.nn.Conv2d(64, 32, 3, 1, 1)
        self.activ8_1 = snntorch.Leaky(beta=beta)
        self.layer8_2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.activ8_2 = snntorch.Leaky(beta=beta)
        self.unpool8 = torch.nn.ConvTranspose2d(32, 16, 2, 2)

        self.layer9_1 = torch.nn.Conv2d(32, 16, 3, 1, 1)
        self.activ9_1 = snntorch.Leaky(beta=beta)
        self.layer9_2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.activ9_2 = snntorch.Leaky(beta=beta)
        self.layer9_3 = torch.nn.Conv2d(16, 1, 1)
        self.activ9_3 = snntorch.Leaky(beta=beta)

        self.numSteps = numSteps

    def forward(self, batchInput):
        memActiv1_1 = self.activ1_1.init_leaky()
        memActiv1_2 = self.activ1_2.init_leaky()
        memActiv2_1 = self.activ2_1.init_leaky()
        memActiv2_2 = self.activ2_2.init_leaky()
        memActiv3_1 = self.activ3_1.init_leaky()
        memActiv3_2 = self.activ3_2.init_leaky()
        memActiv4_1 = self.activ4_1.init_leaky()
        memActiv4_2 = self.activ4_2.init_leaky()
        memActiv5_1 = self.activ5_1.init_leaky()
        memActiv5_2 = self.activ5_2.init_leaky()
        memActiv6_1 = self.activ6_1.init_leaky()
        memActiv6_2 = self.activ6_2.init_leaky()
        memActiv7_1 = self.activ7_1.init_leaky()
        memActiv7_2 = self.activ7_2.init_leaky()
        memActiv8_1 = self.activ8_1.init_leaky()
        memActiv8_2 = self.activ8_2.init_leaky()
        memActiv9_1 = self.activ9_1.init_leaky()
        memActiv9_2 = self.activ9_2.init_leaky()
        memOutput = self.activ9_3.init_leaky()

        spikeOutputRec = []
        memOutputRec = []

        for step in range(self.numSteps):
            batch = self.layer1_1(batchInput)
            batch, memActiv1_1 = self.activ1_1(batch, memActiv1_1)
            batch = self.layer1_2(batch)
            batchRes1, memActiv1_2 = self.activ1_2(batch, memActiv1_2)
            batch = self.pool1(batchRes1)

            batch = self.layer2_1(batch)
            batch, memActiv2_1 = self.activ2_1(batch, memActiv2_1)
            batch = self.layer2_2(batch)
            batchRes2, memActiv2_2 = self.activ2_2(batch, memActiv2_2)
            batch = self.pool2(batchRes2)

            batch = self.layer3_1(batch)
            batch, memActiv3_1 = self.activ3_1(batch, memActiv3_1)
            batch = self.layer3_2(batch)
            batchRes3, memActiv3_2 = self.activ3_2(batch, memActiv3_2)
            batch = self.pool3(batchRes3)

            batch = self.layer4_1(batch)
            batch, memActiv4_1 = self.activ4_1(batch, memActiv4_1)
            batch = self.layer4_2(batch)
            batchRes4, memActiv4_2 = self.activ4_2(batch, memActiv4_2)
            batch = self.pool4(batchRes4)

            batch = self.layer5_1(batch)
            batch, memActiv5_1 = self.activ5_1(batch, memActiv5_1)
            batch = self.layer5_2(batch)
            batch, memActiv5_2 = self.activ5_2(batch, memActiv5_2)
            batch = self.unpool5(batch)

            batch = self.residualConnection(batchRes4, batch)
            batch = self.layer6_1(batch)
            batch, memActiv6_1 = self.activ6_1(batch, memActiv6_1)
            batch = self.layer6_2(batch)
            batch, memActiv6_2 = self.activ6_2(batch, memActiv6_2)
            batch = self.unpool6(batch)

            batch = self.residualConnection(batchRes3, batch)
            batch = self.layer7_1(batch)
            batch, memActiv7_1 = self.activ7_1(batch, memActiv7_1)
            batch = self.layer7_2(batch)
            batch, memActiv7_2 = self.activ7_2(batch, memActiv7_2)
            batch = self.unpool7(batch)

            batch = self.residualConnection(batchRes2, batch)
            batch = self.layer8_1(batch)
            batch, memActiv8_1 = self.activ8_1(batch, memActiv8_1)
            batch = self.layer8_2(batch)
            batch, memActiv8_2 = self.activ8_2(batch, memActiv8_2)
            batch = self.unpool8(batch)

            batch = self.residualConnection(batchRes1, batch)
            batch = self.layer9_1(batch)
            batch, memActiv9_1 = self.activ9_1(batch, memActiv9_1)
            batch = self.layer9_2(batch)
            batch, memActiv9_2 = self.activ9_2(batch, memActiv9_2)
            batchOutput = self.layer9_3(batch)
            spikeOutput, memOutput = self.activ9_3(batchOutput, memOutput)

            spikeOutputRec.append(spikeOutput)
            memOutputRec.append(memOutput)

        return spikeOutputRec, memOutputRec

    def residualConnection(self, tensorSource, tensorTarget):
        return torch.cat([tensorSource, tensorTarget], 1)
