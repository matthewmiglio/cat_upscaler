import random
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 1. Define the Dataset Class
class SRDataset(Dataset):
    def __init__(self, dataset_name, transform=None):
        datatset_dir = os.path.join(os.getcwd(), 'datasets',dataset_name)
        self.lr_dir = os.path.join(datatset_dir, "LR")
        self.hr_dir = os.path.join(datatset_dir, "HR")
        self.lr_images = os.listdir(self.lr_dir)
        hr_images = os.listdir(self.hr_dir)
        self.transform = transform
        for image_name in self.lr_images:
            if image_name not in hr_images:
                print("Warning: HR image not found for", image_name)
        for image_name in hr_images:
            if image_name not in self.lr_images:
                print("Warning: LR image not found for", image_name)

    def get_dataset_size(self):
        return len(self.lr_images)

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        try:
            this_image_name = self.lr_images[idx]
            lr_image_path = os.path.join(self.lr_dir, this_image_name)
            hr_image_path = os.path.join(self.hr_dir, this_image_name)

            lr_image = Image.open(lr_image_path).convert("RGB")
            hr_image = Image.open(hr_image_path).convert("RGB")

            if self.transform:
                lr_image = self.transform(lr_image)
                hr_image = self.transform(hr_image)

            return lr_image, hr_image
        except Exception as e:
            print("Error loading image:", e)
            return self.__getitem__(random.randing(0, len(self.lr_images)))


# 2. Define the Model (SRCNN or a simple CNN)
class SuperResModel(nn.Module):
    def __init__(self):
        super(SuperResModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# 3. Training Loop with Increased Verbosity
def train(model, dataloaders, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_batches = len(dataloaders)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}] started. Training...")

        for batch_idx, (lr_images, hr_images) in enumerate(dataloaders):
            start_batch_time = time.time()

            # Print batch index and current batch size
            print(
                f"  Processing batch {batch_idx+1}/{total_batches} (batch size: {lr_images.size(0)})",
                end="\r",
            )

            # Move data to GPU if available
            if torch.cuda.is_available():
                lr_images = lr_images.cuda()
                hr_images = hr_images.cuda()

            optimizer.zero_grad()
            output = model(lr_images)
            loss = criterion(output, hr_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print feedback every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(
                    f"    \nBatch {batch_idx+1}/{total_batches} - Loss: {avg_loss:.4f} | Time: {time.time() - start_batch_time:.2f} seconds"
                )

        avg_loss = running_loss / total_batches
        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] completed. Average loss: {avg_loss:.4f} | Time: {time.time() - start_epoch_time:.2f} seconds\n"
        )


# 4. Export Model to ONNX with Verbosity
def export_to_onnx(unique_model_name, model, dummy_input):
    onnx_filename = f"{unique_model_name}.onnx"
    print(f"Exporting model to ONNX format... ({onnx_filename})")
    torch.onnx.export(model, dummy_input, onnx_filename, verbose=True)
    print(f"Model successfully exported to {onnx_filename}")


def get_date_readable():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def run_train_script(dataset_name, epochs, image_height, image_width, batch_size=2):
    train_start_time = time.time()

    # Define the transform to normalize images
    transform = transforms.Compose([transforms.ToTensor()])

    # Create dataset and dataloader
    dataset = SRDataset(dataset_name, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize model
    model = SuperResModel()

    if torch.cuda.is_available():
        print("CUDA is available. Moving model to GPU.")
        model = model.cuda()  # Move model to GPU if available
    else:
        print("CUDA is not available. Using CPU.")

    # Train model
    train(model, dataloader, num_epochs=epochs, lr=0.001)

    # Export model to ONNX
    dummy_input = (
        torch.randn(1, 3, image_width, image_height).cuda()
        if torch.cuda.is_available()
        else torch.randn(1, 3, image_width, image_height)
    )
    this_unique_model_name = f"{dataset_name}_{get_date_readable()}_epochs_{epochs}"
    export_to_onnx(this_unique_model_name, model, dummy_input)

    train_finish_time = time.time()
    train_time_taken_readable = str(
        datetime.timedelta(seconds=train_finish_time - train_start_time)
    )
    print(
        f"\n\Trained {epochs} epochs\non {dataset.get_dataset_size()} images\nof size {image_width}x{image_height}\nin {train_time_taken_readable}."
    )


# 5. Main function to execute the training and export
if __name__ == "__main__":
    for epochs in [1,5, 10, 15, 20]:
        dataset_name = "downscale3_01_05"
        image_height, image_width = 640,640
        run_train_script(dataset_name, epochs, image_height, image_width)
