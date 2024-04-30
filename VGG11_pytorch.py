import torch
import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Block 5
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Classification layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.relu9 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu10 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(x)

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool4(x)

        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.relu9(self.fc1(x))
        x = self.relu10(self.fc2(x))
        x = self.fc3(x)

        return x