import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


# Define the model
class UpscalingCNN(nn.Module):
    def __init__(self):
        super(UpscalingCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# Custom dataset for loading low and high-resolution images
class ImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.lr_dir = os.path.join(dataset_dir,'LR')
        self.hr_dir = os.path.join(dataset_dir,'HR')
        self.lr_images = sorted(os.listdir(self.lr_dir))
        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(os.path.join(self.lr_dir, self.lr_images[idx]))
        hr_image = Image.open(os.path.join(self.hr_dir, self.hr_images[idx]))

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


def train(dataset_dir):
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load dataset
    dataset = ImageDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UpscalingCNN().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.cuda(), hr.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(lr)

            # Calculate loss
            loss = criterion(outputs, hr)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

        # Optionally, save the model after every epoch
        torch.save(model.state_dict(), "upscaling_model.pth")


if __name__ == "__main__":
    dataset_dir = r'C:\Users\matmi\Desktop\my files\my programs\upscaler_train\dataset'
    train(dataset_dir)
