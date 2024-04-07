import torch
import torch.nn as nn

# Binary Cross Entropy for ISBI-2012 dataset
binary_loss_object = nn.BCELoss()

# Cross Entropy Loss for Oxford-IIIT dataset
cross_entropy_loss_object = nn.CrossEntropyLoss()