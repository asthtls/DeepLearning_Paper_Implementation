import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import argparse

from loss_torch import binary_loss_object
from model_torch import UNET_TORCH_ISBI_2012

# normalize isbi-2012 data
def normalize_isbi_2012(input_image, mask_labels):
    # 0~ 255 -> 0.0 ~ 1.0
    input_image = input_image / 255
    mask_labels = mask_labels / 255

    # set label to binary
    mask_labels[mask_labels > 0.5] = 1
    mask_labels[mask_labels <= 0.5] = 0

    return input_image, mask_labels

class ISBI2012Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image, mask = normalize_isbi_2012(image, mask)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = display_list[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

def display_and_save(display_list, epoch):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = display_list[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.savefig(f'epoch {epoch}.jpg')

def create_mask(pred_mask):
    pred_mask = torch.where(pred_mask > 0.5, 1, 0)
    return pred_mask.squeeze().cpu().numpy()

def show_predictions(model, sample_image, sample_mask, device):
    model.eval()
    with torch.no_grad():
        pred_mask = model(sample_image.unsqueeze(0).to(device))
        display([sample_image.squeeze().cpu().numpy(), sample_mask.squeeze().cpu().numpy(), create_mask(pred_mask)])

def save_predictions(epoch, model, sample_image, sample_mask, device):
    model.eval()
    with torch.no_grad():
        pred_mask = model(sample_image.unsqueeze(0).to(device))
        display_and_save([sample_image.squeeze().cpu().numpy(), sample_mask.squeeze().cpu().numpy(), create_mask(pred_mask)], epoch)

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define data augmentation transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=0.05),
        transforms.RandomHorizontalFlip(),
    ])


    # Create dataset and dataloader
    train_dataset = ISBI2012Dataset(image_dir='./isbi_2012/preprocessed/train_imgs',
                                    mask_dir='./isbi_2012/preprocessed/train_labels',
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Get a sample image and mask for visualization
    sample_image, sample_mask = train_dataset[0]

    # Display sample image and mask
    display([sample_image, sample_mask])

    # Create ISBI-2012 UNET model
    unet_model = UNET_TORCH_ISBI_2012(args.num_classes).to(device)

    # Show prediction before training
    show_predictions(unet_model, sample_image, sample_mask, device)

    # Set optimizer and loss function
    optimizer = optim.Adam(unet_model.parameters(), lr=args.learning_rate)
    criterion = binary_loss_object

    # Check if checkpoint path exists
    if not os.path.exists(args.checkpoint_path.split('/')[0]):
        os.mkdir(args.checkpoint_path.split('/')[0])

    # Restore latest checkpoint
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'{args.checkpoint_path} checkpoint is restored!')

    # Start training
    for epoch in range(args.num_epochs):
        unet_model.train()
        for batch_images, batch_masks in train_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()
            outputs = unet_model(batch_images)
            loss = criterion(outputs, batch_masks)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, args.checkpoint_path)

        # Save predictions
        save_predictions(epoch + 1, unet_model, sample_image, sample_mask, device)
        print(f'\nepoch after predict sample: {epoch + 1}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--checkpoint_path', default='saved_model_isbi_2012/unet_model.pth', type=str,
                        help='path to a directory to save model checkpoints during training')
    parser.add_argument('--num_epochs', default=5, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_classes', default=1, type=int, help='number of prediction classes')
    args = parser.parse_args()
    if torch.cuda.is_available():
        # GPU 메모리 증가 허용 설정
        torch.cuda.set_per_process_memory_fraction(0.7)  # 80%의 GPU 메모리 사용 설정
        device = torch.device('cuda')
        print('GPU is available. Training on GPU.')
    else:
        device = torch.device('cpu')
        print('GPU is not available. Training on CPU.')
        
    main(args)