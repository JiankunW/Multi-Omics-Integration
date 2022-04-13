import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import linalg as LA
from torch import nn


def feature_selection(X, y):
    data_label1 = np.asarray([X[i] for i in range(len(y)) if y[i] == 1])
    data_label0 = np.asarray([X[i] for i in range(len(y)) if y[i] == 0])
    p = ttest_ind(data_label1, data_label0)[1]
    keep_ttest_index = np.argsort(p)[0:3000]  # np.where(p < .001)[0]
    return keep_ttest_index


# def load_data(path):
#     data = pd.read_csv(path, delimiter=",", index_col=0)
#     cols = data.columns.tolist()
#     data = np.log1p(data)
#     data.loc[:, "var"] = data.loc[:, cols].var(axis=1)
#     drop_index = data[data["var"] < 0].index.tolist()
#     data.drop(index=drop_index, inplace=True)
#     X = data[cols]

#     return X


def SVM(X_train, y_train, X_test, y_test):

    clf = svm.SVC(kernel="linear", probability=True).fit(X_train, y_train)
    # clf = RandomForestClassifier(n_estimators=150, random_state=0,min_samples_split=5).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]

    y_pred_bin = np.copy(y_pred)
    y_pred_bin[y_pred_bin < 0.5] = 0
    y_pred_bin[y_pred_bin >= 0.5] = 1

    # return (
    #     roc_auc_score(y_test, y_pred),
    #     accuracy_score(y_test, y_pred_bin),
    #     f1_score(y_test, y_pred_bin),
    # )
    return roc_auc_score(y_test, y_pred)


def SVM_multiclass(X_train, y_train, X_test, y_test):

    clf = svm.SVC(kernel="linear", probability=True).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    return roc_auc_score(y_test, y_pred, multi_class="ovr")


def prediction(mRNA_value, GAN_epoch, labels):
    X = np.array(mRNA_value).astype(float)

    trial = 50
    AUC_all = []
    dim_labels = labels.shape[1]

    for col in range(dim_labels):
        y = labels[:, col]
        cls_counts = len(np.unique(y))
        AUC = []
        for i in range(trial):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i
            )
            target = feature_selection(X_train, y_train)
            X_train = X_train[:, target]
            X_test = X_test[:, target]
            if cls_counts == 2:
                auc = SVM(X_train, y_train, X_test, y_test)
            elif cls_counts > 2:
                auc = SVM_multiclass(X_train, y_train, X_test, y_test)
            AUC.append(auc)
        AUC_all.append(sum(AUC) / trial)

    AUC_final = sum(AUC_all) / dim_labels

    if GAN_epoch == 0:
        print("AUC for real data:", AUC_final)
    else:
        print("AUC for generated data at epoch", GAN_epoch, ":", AUC_final)

    return AUC_final


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 768),
            # nn.BatchNorm1d(768),
            # nn.ReLU(),
            # nn.Linear(768, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            nn.Linear(1024, n_input),
        )

    def forward(self, x):
        output = self.model(x)

        return output


def omics2(
    update: int, mRNA_file: str, miRNA_file: str, adj_file: str, label_file: str, device
) -> None:

    # Set how floating-point errors are handled.
    np.seterr(divide="ignore", invalid="ignore")

    print("mRNA_" + str(update))

    # mRNA = load_data(mRNA_file)
    mRNA = pd.read_csv(mRNA_file, delimiter=",", index_col=0)
    # mRNA = np.log1p(mRNA)
    miRNA = pd.read_csv(miRNA_file, delimiter=",", index_col=0)
    # miRNA = np.log1p(miRNA)
    # miRNA = load_data(miRNA_file)
    adj = pd.read_csv(adj_file, index_col=0)
    _, x_ind, y_ind = np.intersect1d(mRNA.columns, miRNA.columns, return_indices=True)
    _, x_ind1, y_ind1 = np.intersect1d(miRNA.index, adj.columns, return_indices=True)
    _, x_ind2, y_ind2 = np.intersect1d(mRNA.index, adj.index, return_indices=True)
    mRNA = mRNA.iloc[x_ind2, x_ind]
    miRNA = miRNA.iloc[x_ind1, y_ind]
    adj = adj.iloc[y_ind2, y_ind1]
    mRNA = mRNA.fillna(0)
    miRNA = miRNA.fillna(0)

    adj[adj == 1] = -1
    adj[adj == 0] = 1

    labels = pd.read_csv(label_file, delimiter=",", index_col=0)
    _, x_ind3, y_ind3 = np.intersect1d(mRNA.columns, labels.index, return_indices=True)
    mRNA = mRNA.iloc[:, x_ind3]
    miRNA = miRNA.iloc[:, x_ind3]
    labels = labels.iloc[y_ind3, :]
    labels = np.array(labels).astype(np.float32)

    sample_name = mRNA.columns
    feature_name = mRNA.index

    mRNA = np.array(mRNA).transpose().astype(np.float32)  # nxp
    miRNA = np.array(miRNA).astype(np.float32)  # mxn
    adj = np.array(adj).transpose().astype(np.float32)  # mxp

    X_0 = torch.from_numpy(mRNA).to(device)

    if update > 1:
        mRNA_file_name = "mRNA_BRCA" + str(update - 1) + ".csv"
        miRNA_file_name = "miRNA_BRCA" + str(update - 1) + ".csv"

        mRNA = pd.read_csv(mRNA_file_name, delimiter=",", index_col=0)
        mRNA = np.array(mRNA).astype(np.float32)
        miRNA = pd.read_csv(miRNA_file_name, delimiter=",", index_col=0)
        miRNA = np.array(miRNA).astype(np.float32).transpose()

    n_input_mRNA = np.size(mRNA, 1)
    sample_size = np.size(mRNA, 0)

    C = np.sqrt(np.outer(np.sum(np.absolute(adj), 0), np.sum(np.absolute(adj), 1)))
    adj = np.divide(adj, C.transpose())

    mRNA_train_data = torch.from_numpy(mRNA)
    adj = torch.from_numpy(adj)

    mRNA_train_labels = torch.zeros(sample_size)

    mRNA_train_set = [
        (mRNA_train_data[i], mRNA_train_labels[i]) for i in range(sample_size)
    ]

    batch_size = sample_size
    mRNA_train_loader = torch.utils.data.DataLoader(
        mRNA_train_set, batch_size=batch_size, shuffle=False
    )

    discriminator = Discriminator(n_input_mRNA).to(device)
    generator = Generator(n_input_mRNA).to(device)

    lr_D = 5e-6
    lr_G = 5e-5
    num_epochs = 10000
    critic_ite = 5
    weight_clip = 0.01
    output_file = "best_mRNA.txt"

    optimizer_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
    optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr_G)
    best = None
    dloss_epoch = []
    gloss_epoch = []

    for epoch in range(num_epochs):
        dloss_batch = []
        gloss_batch = []
        for n, (real_samples, _) in enumerate(mRNA_train_loader):
            miRNA_train_data = miRNA[:, n * batch_size : (n + 1) * batch_size]
            miRNA_train_data = torch.from_numpy(miRNA_train_data)
            latent_value = torch.matmul(miRNA_train_data.t(), adj)
            real_samples = real_samples.to(device)
            latent_value = latent_value.to(device)

            # training the discriminator
            dloss_batch_list = []
            for _ in range(critic_ite):
                generated_samples = generator(latent_value)

                discriminator.zero_grad()
                output_discriminator_real = discriminator(real_samples)
                output_discriminator_fake = discriminator(generated_samples)

                loss_discriminator = torch.mean(output_discriminator_fake) - torch.mean(
                    output_discriminator_real
                )
                loss_discriminator.backward(retain_graph=True)
                optimizer_discriminator.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

                dloss_batch_list.append(loss_discriminator.item())
            dloss_batch.append(sum(dloss_batch_list) / len(dloss_batch_list))

            # Training the generator
            generator.zero_grad()
            output_discriminator_fake = discriminator(generated_samples)
            loss_generator = -torch.mean(output_discriminator_fake) + 0.01 * LA.norm(
                (X_0 - generated_samples), 2
            )

            loss_generator.backward()
            optimizer_generator.step()
            gloss_batch.append(loss_generator.item())

        dloss_cur = sum(dloss_batch) / len(dloss_batch)
        gloss_cur = sum(gloss_batch) / len(gloss_batch)
        dloss_epoch.append(dloss_cur)
        gloss_epoch.append(gloss_cur)
        if epoch % 100 == 99:
            print(
                f"mRNA [{epoch}/{num_epochs}] G loss: {dloss_cur}, D loss: {gloss_cur}"
            )

        filename = "mRNA_BRCA" + str(update) + ".csv"
        if epoch == 0:
            auc = prediction(real_samples.cpu().detach().numpy(), epoch, labels)
        elif epoch % 300 == 299:
            auc = prediction(generated_samples.cpu().detach().numpy(), epoch, labels)
            if best is None:
                best = np.mean(auc)
                best_epoch = epoch
                dd = pd.DataFrame(
                    generated_samples.cpu().detach().numpy(),
                    index=sample_name,
                    columns=feature_name,
                )
            elif np.mean(auc) > np.mean(best):
                best = np.mean(auc)
                best_epoch = epoch
                dd = pd.DataFrame(
                    generated_samples.cpu().detach().numpy(),
                    index=sample_name,
                    columns=feature_name,
                )

    print(best, file=open(output_file, "a"))
    print(f"Best auc score: {best} on the epoch {best_epoch}/{num_epochs}")
    dd.to_csv(filename)

    # save loss curve
    plt.figure()
    plt.plot(dloss_epoch, "g", label="D loss")
    plt.plot(gloss_epoch, "r", label="G loss")
    plt.legend()
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    plt.ylim((-10, 20))
    plt.title("mRNAGAN Loss over epochs at Update " + str(update))
    plt.savefig("plots/mloss" + str(update) + ".png")
