from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, ParameterGrid
from torch import nn



def vanillaMulandAdd(
    mRNA_input: np.ndarray, miRNA_input: np.ndarray, adj_input: np.ndarray
) -> List[np.ndarray]:
    latent_values_1 = np.matmul(mRNA_input, adj_input)
    latent_values_2 = np.matmul(miRNA_input, adj_input.transpose())
    mRNA_new = mRNA_input + latent_values_2
    miRNA_new = miRNA_input + latent_values_1
    return [mRNA_new, miRNA_new]


class Addnet(nn.Module):
    def __init__(self, nfeats1, nfeats2):
        """
        x = input1 x (adj * W) + input2
        output = MLP(x)

        x: matrix mul
        * element-wise mul
        input1: n by nfeats1
        input2: n by nfeats2
        adj: nfeats1 by nfeats2

        parameters:
        MLP weights, W
        """
        super().__init__()
        self.register_parameter(name="W", param=nn.Parameter(torch.rand(nfeats1, nfeats2)))
        if nfeats2 > 1000:
            self.mlp = nn.Sequential(
                nn.Linear(nfeats2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                # nn.Linear(1024, 2048),
                # nn.BatchNorm1d(2048),
                # nn.ReLU(),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        elif nfeats2 > 64:
            self.mlp = nn.Sequential(
                nn.Linear(nfeats2, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                # nn.Linear(256, 64),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(nfeats2, 4),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

    def forward(self, input1, input2, adj):
        x = torch.matmul(input1, adj * self.W) + input2
        x = self.mlp(x)
        return x


class Dataset_add(torch.utils.data.Dataset):
    def __init__(self, input1, input2, labels):
        self.input1 = input1
        self.input2 = input2
        self.labels = labels

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, index):
        return (
            self.input1[index],
            self.input2[index],
            self.labels[index],
        )

class trainADD(object):
    def __init__(self, nfeats1, nfeats2, hyperparams, device, dataset_name):
        self.hyperparams = hyperparams
        self.model = Addnet(nfeats1, nfeats2).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=hyperparams["lr"]
        )
        self.loss_fn = nn.BCELoss()
        self.device = device
        self.dataset_name = dataset_name

    def train(
        self, input1: np.ndarray, input2: np.ndarray, adj: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Train a mul matrix

        Args:
            input1 (np.ndarray): input mRNA data (n by nfeats1)
            input2 (np.ndarray): input miRNA data (n by nfeats2)
            adj (np.ndarray): input adjacency matrix (nfeats1 by nfeats2)
            labels (np.ndarray): label vector
            device: gpu or cpu

        Returns:
            the mul matrix (np.ndarray): nfeat1 by nfeat2
        """
        self.model.train()
        input1 = torch.from_numpy(input1).to(self.device)
        input2 = torch.from_numpy(input2).to(self.device)
        adj = torch.from_numpy(adj).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        training_set = Dataset_add(input1, input2, labels)
        training_generator = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
            num_workers=self.hyperparams["num_workers"],
        )

        losses = []
        auc_rocs = []
        num_epochs = self.hyperparams["num_epochs"]
        for epoch in range(num_epochs):
            for (input1_samples, input2_samples, label_samples) in training_generator:
                output = self.model(input1_samples, input2_samples, adj)
                loss = self.loss_fn(output, label_samples.reshape(-1, 1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # accuracy
                auc = roc_auc_score(label_samples.detach().cpu().numpy(), output.detach().cpu().numpy())

            if epoch % 400 == 0:
                losses.append(loss)
                auc_rocs.append(auc)
                # print("epoch {}\tloss : {}\t auc : {}".format(epoch, loss, auc))
        # plt.figure()
        # plt.plot(losses)
        # plt.title('Loss vs Epochs')
        # plt.xlabel('Epochs')
        # plt.ylabel('loss')
        # plt.savefig("plots/" + str(self.dataset_name) + "_addnetloss.png")

    def evaluate(self, input1: np.ndarray, input2: np.ndarray, adj: np.ndarray, labels: np.ndarray):
        self.model.eval()
        input1 = torch.from_numpy(input1).to(self.device)
        input2 = torch.from_numpy(input2).to(self.device)
        adj = torch.from_numpy(adj).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        output = self.model(input1, input2, adj)
        auc = roc_auc_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        return auc

def omicsAdd(
    mRNA_input: np.ndarray,
    miRNA_input: np.ndarray,
    adj_input: np.ndarray,
    labels: np.ndarray,
    device,
    dataset_name: str,
) -> List[np.ndarray]:
    """Two omics input data are integrated through MLP. Output are
    the list of two synthetic omics data.

    Args:
        mRNA_input (np.ndarray): input mRNA data (n by nfeats1)
        miRNA_input (np.ndarray): input miRNA data (n by nfeats2)
        adj_input (np.ndarray): input adjacency matrix (nfeats1 by nfeats2)
        labels (np.ndarray): labels for samples

    Returns:
        List[np.ndarray]: List of two ndarray items, the final synthetic mRNA
        expression and the final synthesic miRNA expression
    """
    assert np.size(mRNA_input, 0) == np.size(miRNA_input, 0)

    dim_labels = labels.shape[1]
    n_mRNA_feats = np.size(mRNA_input, 1)
    n_miRNA_feats = np.size(miRNA_input, 1)

    AUC_m = []
    AUC_mi = []
    trial = 50
    parameters = {
        'lr': [0.0001, 0.001, 0.005, 0.01, 0.05],
        "num_epochs":[400, 800, 1600, 2000, 4000, 8000, 16000],
    }
    best_m = 0
    best_mi = 0
    for d in list(ParameterGrid(parameters)):
        for col in range(dim_labels):
            # print(f"dim labels {col}/{dim_labels}")
            y = labels[:, col]
            for i in range(trial):
                # print(f"trial {i}/{trial}")
                X_train, X_test, y_train, y_test = train_test_split(
                    np.concatenate((mRNA_input, miRNA_input), axis=1),
                    y,
                    test_size=0.2,
                    random_state=i,
                )
                sample_size = X_train.shape[1]
                hypers_mRNA = {
                    "lr": d["lr"],
                    "num_epochs": d["num_epochs"],
                    "batch_size": sample_size,
                    "num_workers": 0,
                }
                hypers_miRNA = {
                    "lr": d["lr"],
                    "num_epochs": d["num_epochs"],
                    "batch_size": sample_size,
                    "num_workers": 0,
                }
                mRNA_train, miRNA_train = (
                    X_train[:, :n_mRNA_feats],
                    X_train[:, n_mRNA_feats:],
                )
                mRNA_test, miRNA_test = X_test[:, :n_mRNA_feats], X_test[:, n_mRNA_feats:]

                # train syntesis mRNA
                model_m = trainADD(n_miRNA_feats, n_mRNA_feats, hypers_mRNA, device, dataset_name)
                model_m.train(miRNA_train, mRNA_train, adj_input.transpose(), y_train)

                # test syntesis mRNA
                auc_m = model_m.evaluate(miRNA_test, mRNA_test, adj_input.transpose(), y_test)
                # print(f"dim {col}/{dim_labels}\t trial {i}/{trial}\t syntesis mRNA test auc {auc_m}")

                # train syntesis miRNA
                model_mi = trainADD(n_mRNA_feats, n_miRNA_feats, hypers_miRNA, device, dataset_name)
                model_mi.train(mRNA_train, miRNA_train, adj_input, y_train)

                # test syntesis miRNA
                auc_mi = model_mi.evaluate(mRNA_test, miRNA_test, adj_input, y_test)
                # print(f"dim {col}/{dim_labels}\t trial {i}/{trial}\t syntesis miRNA test auc {auc_m}")

                AUC_m.append(auc_m)
                AUC_mi.append(auc_mi)
        auc_final_m = sum(AUC_m) / len(AUC_m)
        auc_final_mi = sum(AUC_mi) / len(AUC_mi)
        print(d)
        print(f"AUC score for synthetic mRNA: {auc_final_m}")
        print(f"AUC score for synthetic miRNA: {auc_final_mi}")
        if auc_final_m > best_m:
            best_m = auc_final_m
            best_d_m = d
        if auc_final_mi > best_mi:
            best_mi = auc_final_mi
            best_d_mi = d

    print("\n")
    print(f"Best AUC score for synthetic mRNA: {best_m}\t params {best_d_m}")
    print(f"Best AUC score for synthetic miRNA: {best_mi}\t params {best_d_mi}")
    return auc_final_m, auc_final_mi
