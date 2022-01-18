import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)

model_path= './pre_trained_weights.pt'


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=5, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.Conv2d(4, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(8, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=32)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=4, padding=1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 8, kernel_size=7, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=5, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# make model instance
model = Autoencoder()

# load model weights
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


def normalize(data):
    data_max = data.max()
    data_min = data.min()
    if data_min != data_max:
        data = ((data - data_min) / (data_max - data_min))
    return data


# tasks on full dataset
dataset = torchvision.datasets.EMNIST(
    root='../data',
    split='bymerge',
    train=False,
    download=True
)
x = np.expand_dims(dataset.data.numpy(), axis=1)
x = normalize(x)
x = torch.FloatTensor(x)

# fill with reference images
reference_images = torch.empty((len(dataset.classes), *x[0].size()))
samples_to_remove = []
for idx in range(len(dataset.classes)):
    sample_idx = np.where(dataset.targets == idx)[0][0]    # select one sample from class idx
    sample = x[sample_idx]  # select sample
    reference_images[idx] = sample
    samples_to_remove.append(sample_idx)

samples_to_remove = sorted(samples_to_remove, reverse=True)
for idx in samples_to_remove:
    x = torch.cat((x[:,:idx], x[:,idx+1:]), axis=1)

# process reference images
with torch.no_grad():
    refence_embedding = model.encoder(reference_images)
    refence_embedding_flat = refence_embedding.view(refence_embedding.size(0), -1)

# a lot of first samples will be duplicates from reference images
for idx, sample in enumerate(x):
    pass
    # TODO process sample embedding
    # TODO calculate cosine distance
    # TODO select correct class
    # TODO plot results (images of reference samples and closest ones in

