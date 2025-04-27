import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils

# Load and prep real data
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)

# Hyperparameters
batch_size = 64
lr = 0.0002
beta1 = 0.5
nz = 100 # Size of the latent z vector (input noise)
epochs = 10

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

# Initialise models and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # make sure your device has GPU
# mine doesn't so 
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))

# Loss function (Binary Cross Entropy)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for i, (data, _) in enumerate(dataloader, 0):
        # Train Discriminator
        netD.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1.0, device = device)

        output = netD(real_data)
        errD_real = criterion(output.view(-1), label)
        errD_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device = device)
        fake_data = netG(noise)
        label.fill_(0.0)
        output = netD(fake_data.detach())
        errD_fake = criterion(output.view(-1), label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(1.0)
        output = netD(fake_data)
        errG = criterion(output.view(-1), label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch} / {epochs}][{i} / {len(dataloader)}] Loss_D : {errD.item()} Loss_G : {errG.item()}')

# Save generated images after each epoch
vutils.save_image(fake_data.data, f'output/fake_samples_epoch_{epoch}.png', normalize = True)

