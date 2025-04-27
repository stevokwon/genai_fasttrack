import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f'Using device : {device}')

# Hyperparameters
batch_size = 64
lrG = 0.0002
lrD = 0.0001 # Slightly slower D learning rate
beta1 = 0.5
nz = 100 # Size of the latent z vector (input noise)
epochs = 10

# Create output folder
os.makedirs('output', exist_ok = True)

# Load and prep real data
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(256, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.main(input)
        out = out.mean([2, 3]) # Average over 4x4 spatial map
        return out.view(-1)    # Flatten to (batch_size) 


netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas = (beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas = (beta1, 0.999))

# Fixed noise for consistent visualisation
fixed_noise = torch.randn(64, nz, 1, 1, device = device)

# Helper to visualise generated images
def show_generated_images(generator, epoch):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
    grid = np.transpose(vutils.make_grid(fake, padding = 2, normalize = True), (1, 2, 0))
    plt.figure(figsize = (6, 6))
    plt.axis('off')
    plt.title(f'Generated Images at Epoch {epoch}')
    plt.imshow(grid)
    plt.savefig(f'output/fake_samples_epoch_{epoch}.png')
    plt.close()
    generator.train()

# Training loop
print(f'Starting Training Loop....')
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()
        real_data = data.to(device)
        batch_size_curr = real_data.size(0) # use dynamic batch size

        # Add tiny noise to real data
        real_data += 0.05 * torch.randn_like(real_data)

        # Real label smoothing (0.8 to 1.0)
        real_label = torch.empty(batch_size_curr, device=device).uniform_(0.8, 1.0)
        output = netD(real_data)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        # Generate fake data
        noise = torch.randn(batch_size_curr, nz, 1, 1, device=device)
        fake = netG(noise)

        fake_label = torch.zeros(batch_size_curr, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        label_for_generator = torch.ones(batch_size_curr, device=device)  # generator tries to fool D
        output = netD(fake)
        errG = criterion(output, label_for_generator)
        errG.backward()
        optimizerG.step()

        # Print every 100 batches
        if i % 100 == 0:
            print(f"[{epoch} / {epochs}][{i} / {len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

    # Save samples after each epoch
    show_generated_images(netG, epoch)

print("Training Complete.")