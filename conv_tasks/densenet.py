import functools
import random

import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BATCH_SIZE = 16
n_classes = 0
n_names = np.array([])
MAX_LEN = 200
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        global n_classes
        global n_names

        super().__init__()
        self.data = fetch_lfw_people(color=True)
        n_classes = self.data.target_names.size
        n_names = self.data.target_names


    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        np_x = self.data.images[idx]
        # np_x = np.expand_dims(np_x, axis=0)
        x = torch.FloatTensor(np_x)
        x = torch.movedim(x, 2, 0)

        np_y = np.zeros((n_classes,))
        np_y[self.data.target[idx]] = 1
        y = torch.FloatTensor(np_y)

        return x, y


dataset = LoadDataset()
devide_by_idx = np.arange(len(dataset))
subset_train_data, subset_test_data = train_test_split(
    devide_by_idx,
    test_size=0.2,
    random_state=11)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class DenseBlock(torch.nn.Module):
    def __init__(self, in_features, num_chains = 4):
        super().__init__()

        self.chains = []
        for i in range(num_chains):
            out_features = (i + 1) * in_features
            self.chains.append(torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(
                    out_features
                ),
                torch.nn.Conv2d(
                    in_channels=out_features,
                    out_channels=in_features,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            ).to(DEVICE))

    def parameters(self):
        return functools.reduce(lambda a, b: a + b, (list(it.parameters()) for it in self.chains))

    def forward(self, x):
        inp = x
        list_out = [x]
        for chain in self.chains:
            out = chain.forward(inp)
            list_out.append(out)
            inp = torch.cat(list_out, dim=1)
        return inp

class TransitionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            torch.nn.AvgPool2d(kernel_size=2,
                               stride=2,
                               padding=0
                               )
        )

    def forward(self, x):
        return self.layers.forward(x)


class Reshape(torch.nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    def forward(self,x ):
        return x.view(self.target_shape)


class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 32
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=num_channels,
                kernel_size=7,
                stride=1,
                padding=1
            ),
            # torch.nn.MaxPool2d(kernel_size=7,
            #                    stride=2 ),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels+4*num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + 4 * num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + 4 * num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + 4 * num_channels, out_features=num_channels),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers.forward(x)


model = DenseNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in tqdm(range(1, 500)):
    plt.clf()

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)

            loss = torch.sum(-y*torch.log(y_prim + 1e-8))
            # Sum dependant on batch size => larger LR
            # Mean independant of batch size => smaller LR

            # y.to('cuda')
            # y.cuda()

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.mean((idx_y == idx_y_prim) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc)

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 3)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plts = []
    c = 0
    for key, value in metrics.items():
        value = gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig('plot_densenet_lfw')
plt.savefig('plot_densenet_lfw')


