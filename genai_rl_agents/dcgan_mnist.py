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
            nn.Conv2d(256, 1, 4, 2, 1, bias = False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

# Initialise models and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # make sure your device has GPU
# mine doesn't so 
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))

# Loss function (Binary Cross Entropy)
criterion = nn.BCELoss()


