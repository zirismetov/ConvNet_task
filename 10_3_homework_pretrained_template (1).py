import torchvision.models
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
MAX_LEN = 200
INPUT_SIZE = 28
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0

class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)

        np_x = np.expand_dims(np_x, axis=0)

        x = torch.FloatTensor(np_x)

        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg11(pretrained=True).features
        self.fc = torch.nn.Linear(
            in_features=10, # TODO change in_features
            out_features=10
        )

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(128, 128)) # upscale for pretrained deep model
        x = x.expand(x.size(0), 3, 128, 128) # repeat grayscale over RGB channels
        #TODO finish code
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            #TODO add confusion matric in tensorboard
            #TODO calculate train and test F1 score
            #TODO add metrics in tensorboard


        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')


    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.show()