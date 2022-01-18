import argparse
import functools
import os
import time

# from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib
import torchvision
import torch.nn.functional
import matplotlib.pyplot as plt
import torch.utils.data
import tensorflow
# import tensorboardX
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import make_grid
from sklearn.datasets import make_blobs
from conv_tasks import tensorboard_utils
# from show_samples import make_grid_with_labels


parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-seq_name', default=f'seq_name', type=str)
parser.add_argument('-is_cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-lr', default=1e-4, type=float)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-epochs', default=10, type=int)
args = parser.parse_args()



BATCH_SIZE = 64
n_classes = 0
n_names = []
weights = []
MAX_LEN = 200
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        global n_classes
        global n_names
        global weights
        self.transform = transform

        super().__init__()
        self.data = fetch_lfw_people(color=True, min_faces_per_person=80)
        n_classes = self.data.target_names.size
        n_names = self.data.target_names

        num_each_classes = {}
        for item in self.data.target_names.tolist():
            num_each_classes[item] = 0

        for t in self.data.target.tolist():
            num_each_classes[self.data.target_names[t]] += 1
        weights = num_each_classes
        # for key, value in num_each_classes.items():
        #     weights.append(value)
        # weights = np.array(weights)







    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        np_x = self.data.images[idx]
        x = torch.FloatTensor(np_x)
        x = torch.permute(x, (2,0,1))


        y = self.data.target[idx]

        if self.transform:
            img = self.transform(x)
            x = img

        return x, y


torchvision_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: torchvision.transforms.functional.invert(x)),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
dataset = LoadDataset(transform=torchvision_transform)
divide_idx = np.arange(len(dataset))
subset_train_data, subset_test_data = train_test_split(
    divide_idx,
    test_size=0.3,
    random_state=11,
    shuffle=None,
    stratify=dataset.data.target)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)



data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
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

class View_Result(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = 0

    def forward(self, x):
        x_out = x
        if self.training:
            # if self.l % 10 == 0:
            inp = torch.nn.functional.adaptive_avg_pool3d(x.detach().cpu(), 3)
            img = make_grid_with_labels(tensor= inp,
                                        labels=n_names)
            show(img, self.l)

            self.l +=1
        return x_out

def quick_show(tensor):
    img = make_grid_with_labels(tensor=tensor,
                                labels=n_names)
    show(img, 0)

class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 32
        num_chains = 4
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
            TransitionLayer(in_features=num_channels+num_chains*num_channels, out_features=num_channels),
            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + num_chains * num_channels, out_features=num_channels),

            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + num_chains * num_channels, out_features=num_channels),

            DenseBlock(in_features=num_channels),
            TransitionLayer(in_features=num_channels + num_chains * num_channels, out_features=num_channels),
            # View_Result(),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Reshape(target_shape=(-1, num_channels)),
            torch.nn.Linear(in_features=num_channels, out_features=n_classes),
        )

    def forward(self, x):
        output = self.layers.forward(x)
        soft = torch.softmax(output, dim=1)
        return soft

model = DenseNet()
model = model.to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)


class_weight_manual_indian=[]
for name in weights:
    class_weight_manual_indian.append(len(dataset.data.data)/(n_classes * weights[name]))
class_weight_manual_indian = torch.FloatTensor(class_weight_manual_indian).to(DEVICE)



def show(img, l = None):
    if not l:
        l = 'check'
    npimg = img.cpu().numpy()
    plt.imsave(f'dense_test_{l}.png',np.transpose(npimg, (1, 2, 0)))

def conf_matrix_f1(y, y_prim):
    conf_matrix = np.zeros((n_classes, n_classes))
    y_prim = np.argmax(y_prim.cpu().detach().numpy(), axis=1)
    y = y.cpu().detach().numpy()
    f1 = []

    for i in range(len(y)):
        conf_matrix[y[i]][y_prim[i]] += 1

    top_10 = {}
    for i in range(n_classes):
        # top_10[i] = conf_matrix[i][i]
        TP = conf_matrix[i][i]
        FP  = np.sum(np.delete(conf_matrix[i,:], i))
        FN  = np.sum(np.delete(conf_matrix[:,i], i))

        score = 2*TP/(2*TP + FP + FN + 1e-9)
        f1.append(score)
    # top_10 = dict(sorted(top_10.items(), key=lambda item: item[1]))
    # top_10 = list(top_10.items())
    # conf_matrix = np.zeros((20, 10))
    # for i in range(10):
    #     for y in range(10):
    #         conf_matrix[i][y] = top_10[-10+y][1]
    #
    # for i in range(10 ,20):
    #     for y in range(10):
    #         conf_matrix[i][y] = top_10[y][1]

    f1 = np.mean(np.array(f1))


    return conf_matrix, f1

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'f1',

    ]:
        metrics[f'{stage}_{metric}'] = []
summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'seq_name/run_name'
)
for epoch in tqdm(range(1, 500)):
    plt.clf()

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        model.train()
        if data_loader == data_loader_test:
            stage = 'test'
            model.eval()
        conf_matrix = []


        for x, y in data_loader:

            # quick_show(x)
            x = x.to(DEVICE)
            y = y.to(DEVICE)


            y_prim = model.forward(x)


            loss = -torch.sum(class_weight_manual_indian[y]*torch.log(y_prim[range(len(x)), y[range(len(x))]] + 1e-8))

            conf_matrix, f1 = conf_matrix_f1(y, y_prim)

            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())# Tensor(0.1) => 0.1f
            metrics_epoch[f'{stage}_f1'].append(f1)


            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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
    plt.savefig('plot_densenet_weighted_lfw')

    metrics_epoch[f'{stage}_loss'] = np.mean(metrics_epoch[f'{stage}_loss'])
    metrics_epoch[f'{stage}_f1'] = np.mean(metrics_epoch[f'{stage}_f1'])


    summary_writer.add_scalar(
        tag=f'{stage}_loss',
        scalar_value=metrics_epoch[f'{stage}_loss'],
        global_step=epoch
    )

    summary_writer.add_scalar(
        tag=f'{stage}_f1',
        scalar_value=metrics_epoch[f'{stage}_f1'],
        global_step=epoch
    )

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict={
            f'{stage}_f1': metrics_epoch[f'{stage}_f1']
        },
        name=args.run_name,
        global_step=epoch
    )

    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest',
               cmap=plt.get_cmap('Greys'))
    plt.xticks(list(range(n_classes)), n_names.tolist())
    plt.yticks(list(range(n_classes)), n_names.tolist())
    for x in range(n_classes):
        for y in range(n_classes):
            plt.annotate(
                str(conf_matrix[x, y]), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white',

            )
    plt.xlabel('True')
    plt.ylabel('Predicted')

    summary_writer.add_figure(
        tag='conf_matrix',
        figure=fig,
        global_step=epoch
    )

    embeddings, classes = make_blobs(n_samples=1000, n_features=128, centers=3)
    summary_writer.add_embedding(
        mat=embeddings,
        metadata=classes.tolist(),
        tag='embeddings',
        global_step=epoch
    )

    summary_writer.flush()

summary_writer.close()

plt.savefig('plot_densenet_weighted_lfw')


