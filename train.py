import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Unet, Discriminator
from ColorizationDataset import LABDataset

"""
This file trains the model
"""

num_epochs = 50
batch_size = 64
learning_rate = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_data_loaders():
    """Initialize the dataset and data loaders."""
    train_l_path = "preprocessed_data/training/L/"
    train_ab_path = "preprocessed_data/training/AB/"
    valid_l_path = "preprocessed_data/validation/L/"
    valid_ab_path = "preprocessed_data/validation/AB/"
    
    train_dataset = LABDataset(train_l_path, train_ab_path)
    valid_dataset = LABDataset(valid_l_path, valid_ab_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
    
    return train_loader, valid_loader

def weights_init_normal(m):
    """Applies custom weight initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def initialize_model():
    """Initialize the U-Net model and transfer it to the GPU if available."""

    # Initialize the model and the discriminator
    generator = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
    discriminator = Discriminator(input_nc=3, ndf=64, n_layers=3)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    return generator, discriminator

def train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, num_epochs, scheduler=None):
    """Train the model with the given parameters."""
    scaler = GradScaler("cuda")

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    train_G_losses = []
    train_D_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_G_loss = 0
        epoch_D_loss = 0
        start_time = time.time()

        for L, AB in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=0, leave=True):
            L, AB = L.to(device), AB.to(device)
            optimizer_D.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                fake_AB = generator(L)

                # Create real and fake input pairs for the discriminator
                real_input = torch.cat((L, AB), dim=1)   # Real pair (L and real AB)
                fake_input = torch.cat((L, fake_AB.detach()), dim=1)  # Fake pair (L and fake AB)

                # Discriminator output for real images
                pred_real = discriminator(real_input)
                # Discriminator output for fake images
                pred_fake = discriminator(fake_input)

                # Labels for real and fake images
                real_labels = torch.ones_like(pred_real)
                fake_labels = torch.zeros_like(pred_fake)

                # Compute discriminator losses
                loss_D_real = criterion_GAN(pred_real, real_labels)
                loss_D_fake = criterion_GAN(pred_fake, fake_labels)
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            # Backpropagate discriminator loss
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            optimizer_G.zero_grad()

            # Generate fake images again for generator update
            with autocast(device_type="cuda"):
                fake_AB = generator(L)
                fake_input = torch.cat((L, fake_AB), dim=1)
                pred_fake = discriminator(fake_input)

                loss_G_GAN = criterion_GAN(pred_fake, real_labels)

                # Generator reconstruction loss
                loss_G_L1 = criterion_L1(fake_AB, AB) * 100 

                # Total generator loss
                loss_G = loss_G_GAN + loss_G_L1
            
            # Backpropagate generator loss
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()

        avg_G_loss = epoch_G_loss / len(train_loader)
        avg_D_loss = epoch_D_loss / len(train_loader)
        
        generator.eval()
        val_loss = 0
        
        with torch.no_grad(), autocast(device_type="cuda"):
            for L, AB in valid_loader:
                L, AB = L.to(device), AB.to(device)
                output = generator(L)
                loss = criterion(output, AB)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        
        train_G_losses.append(avg_G_loss)
        train_D_losses.append(avg_D_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s")
        print(f"Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step()

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")
            print(f"Model checkpoints saved at epoch {epoch + 1}")

    # Save final models
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")

    plot_training_curves(num_epochs, train_G_losses, train_D_losses, val_losses, epoch_times)

def plot_training_curves(num_epochs, train_G_losses, train_D_losses, val_losses, epoch_times):
    """Plot training loss and epoch time curves."""
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 5))

    # Generator Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_G_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()

    # Discriminator Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_D_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    # Validation Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_losses, label='Validation Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    # Plot Time per Epoch
    plt.figure()
    plt.plot(epochs, epoch_times, label='Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time')
    plt.legend()
    plt.savefig('epoch_times.png')
    plt.show()

if __name__ == '__main__':
    train_loader, valid_loader = initialize_data_loaders()
    generator, discriminator = initialize_model()
    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=15, gamma=0.8)
    
    train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, num_epochs)