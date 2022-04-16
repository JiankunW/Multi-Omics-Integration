from typing import Dict, List, MutableMapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import linalg as LA
from torch import nn

from classifier import evaluate


class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        if n_input > 1000:
            self.model = nn.Sequential(
                nn.Linear(n_input, 1024),
                # nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                # nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(256, 1),
                # nn.Sigmoid(),
            )
        elif n_input > 64:
            self.model = nn.Sequential(
                nn.Linear(n_input, 128),
                # nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                # nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(32, 1),
                # nn.Sigmoid(),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(n_input, 8),
                # nn.BatchNorm1d(256),
                nn.ReLU(),
                # nn.Linear(128, 64),
                # nn.BatchNorm1d(64),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(8, 1),
                # nn.Sigmoid(),
            )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        if n_input > 1000:
            self.model = nn.Sequential(
                nn.Linear(n_input, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, n_input),
            )
        elif n_input > 64:
            self.model = nn.Sequential(
                nn.Linear(n_input, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, n_input),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(n_input, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(),
                # nn.Linear(128, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                # nn.Linear(128, 128),
                # nn.BatchNorm1d(128),
                # nn.ReLU(),
                nn.Linear(4, n_input),
            )

    def forward(self, x):
        output = self.model(x)
        return output


class Dataset(torch.utils.data.Dataset):
    def __init__(self, latent_samples, real_samples, original_samples):
        self.latents = latent_samples
        self.reals = real_samples
        self.originals = original_samples

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        return (
            self.latents[index],
            self.reals[index],
            self.originals[index],
        )


def updateFeatures(
    input_X: np.ndarray,
    adj: np.ndarray,
    original_X: np.ndarray,
    target_X: np.ndarray,
    hyperparams: Dict,
    device,
    update_index: int,
    labels: np.ndarray,
    dataset_name: str,
) -> np.ndarray:
    """updating omic1 data supervised by omic2 data

    Args:
        input_X (np.ndarray): omic1 data to be updated
        adj (np.ndarray): adjacency matrix between omic1 and omic2
        original_X (np.ndarray): original omic1 data
        target_X (np.ndarray): omic2 data
        hyperparams (Dict): hyperparameters used in the network
        device (_type_): 'cuda' or 'cpu'
        update_index (int): the index of current update

    Returns:
        np.ndarray: updated omic1 data
    """
    input_X_tensor = torch.from_numpy(input_X).to(device)
    adj_tensor = torch.from_numpy(adj).to(device)
    original_X_tensor = torch.from_numpy(original_X).to(device)
    target_X_tensor = torch.from_numpy(target_X).to(device)

    latent_value_tensor = torch.matmul(input_X_tensor, adj_tensor)
    training_set = Dataset(latent_value_tensor, target_X_tensor, original_X_tensor)
    training_generator = torch.utils.data.DataLoader(
        training_set,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=hyperparams["num_workers"],
    )

    target_feats_num = np.size(target_X, 1)
    discriminator = Discriminator(target_feats_num).to(device)
    generator = Generator(target_feats_num).to(device)
    optimizer_discriminator = torch.optim.RMSprop(
        discriminator.parameters(), lr=hyperparams["lr_D"]
    )
    optimizer_generator = torch.optim.RMSprop(
        generator.parameters(), lr=hyperparams["lr_G"]
    )
    dloss_epoch = []
    gloss_epoch = []
    best_auc = float(0)
    best_epoch = -1
    num_epochs = hyperparams["num_epochs"]
    for epoch in range(num_epochs):
        dloss_batch = []
        gloss_batch = []
        for (latent_samples, real_samples, original_samples,) in training_generator:
            # training the discriminator
            dloss_batch_list = []
            for _ in range(hyperparams["critic_ite"]):
                generated_samples = generator(latent_samples)

                discriminator.zero_grad()
                output_discriminator_fake = discriminator(generated_samples)
                output_discriminator_real = discriminator(real_samples)

                loss_discriminator = torch.mean(output_discriminator_fake) - torch.mean(
                    output_discriminator_real
                )
                loss_discriminator.backward(retain_graph=True)
                optimizer_discriminator.step()

                for p in discriminator.parameters():
                    p.data.clamp_(
                        -hyperparams["weight_clip"], hyperparams["weight_clip"]
                    )

                dloss_batch_list.append(loss_discriminator.item())
            dloss_batch.append(sum(dloss_batch_list) / len(dloss_batch_list))

            # Training the generator
            generator.zero_grad()
            output_discriminator_fake = discriminator(generated_samples)
            loss_generator = -torch.mean(output_discriminator_fake) + hyperparams[
                "alpha"
            ] * LA.norm((original_samples - generated_samples), 2)

            loss_generator.backward()
            optimizer_generator.step()
            gloss_batch.append(loss_generator.item())

        if epoch % 100 == 99:
            dloss_cur = sum(dloss_batch) / len(dloss_batch)
            gloss_cur = sum(gloss_batch) / len(gloss_batch)
            dloss_epoch.append(dloss_cur)
            gloss_epoch.append(gloss_cur)
            # evaluate
            synthetic = generator(latent_value_tensor)
            auc_cur = evaluate(synthetic.detach().cpu().numpy(), labels)
            print(
                f"[{epoch}/{num_epochs}] G loss: {gloss_cur}, D loss: {dloss_cur}, auc: {auc_cur}"
            )
            if auc_cur > best_auc:
                best_auc = auc_cur
                best_epoch = epoch
                best_synthesis = synthetic

    print(f"Best auc score: {best_auc} on the epoch {best_epoch}/{num_epochs}")

    # save loss curve
    plt.figure()
    plt.plot(dloss_epoch, "g", label="D loss")
    plt.plot(gloss_epoch, "r", label="G loss")
    plt.legend()
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    plt.ylim((-1, 2))
    plt.title("miRNAGAN Loss over epochs at Update " + str(update_index))
    plt.savefig("plots/" + str(dataset_name) + "_miloss" + str(update_index) + ".png")

    return best_synthesis.detach().cpu().numpy()

    # if epoch == 0:
    #     auc = prediction(real_samples.cpu().detach().numpy(), epoch, labels)
    # elif epoch % 300 == 299:
    #     auc = prediction(generated_samples.cpu().detach().numpy(), epoch, labels)
    #     if best is None:
    #         best = np.mean(auc)
    #         best_epoch = epoch
    #     elif np.mean(auc) > np.mean(best):
    #         best = np.mean(auc)
    #         best_epoch = epoch


def omicsGAN(
    mRNA_input: np.ndarray,
    miRNA_input: np.ndarray,
    adj_input: np.ndarray,
    labels: np.ndarray,
    total_update: int,
    device,
    dataset_name: str,
) -> List[np.ndarray]:
    """Two omics input data are updated serveral times through GAN. Output are
    the list of two synthetic omics data.

    Args:
        mRNA_input (np.ndarray): input mRNA data (n by nfeats1)
        miRNA_input (np.ndarray): input miRNA data (n by nfeats2)
        adj_input (np.ndarray): input adjacency matrix (nfeats1 by nfeats2)
        labels (np.ndarray): labels for samples
        total_update (int): total number of updates of GANs

    Returns:
        List[np.ndarray]: List of two ndarray items, the final synthetic mRNA
        expression and the final synthesic miRNA expression
    """
    assert np.size(mRNA_input, 0) == np.size(miRNA_input, 0)

    H: MutableMapping[int, List[np.ndarray]] = {
        k: [] for k in range(total_update)
    }  # k->[synthetic mRNA and miRNA after the kth update]
    sample_size = np.size(miRNA_input, 0)

    hypers_mRNA = {
        "lr_D": 5e-6,
        "lr_G": 5e-5,
        "num_epochs": 10000,
        "critic_ite": 5,
        "weight_clip": 0.01,
        "alpha": 0.001,  # L2-norm coefficient
        "batch_size": sample_size,
        "num_workers": 0,
    }
    hypers_miRNA = {
        "lr_D": 5e-6,
        "lr_G": 5e-5,
        "num_epochs": 5000,
        "critic_ite": 5,
        "weight_clip": 0.01,
        "alpha": 0.01,  # L2-norm coefficient
        "batch_size": sample_size,
        "num_workers": 0,
    }

    for update in range(total_update):
        print(f"Update {update}/{total_update-1}")
        mRNA = mRNA_input if update == 0 else H[update - 1][0]
        miRNA = miRNA_input if update == 0 else H[update - 1][1]

        # Generating mRNA features
        print("Generating mRNA features...")
        updated_mRNA = updateFeatures(
            miRNA,
            adj_input.transpose(),
            mRNA_input,
            mRNA,
            hypers_mRNA,
            device,
            update,
            labels,
            dataset_name,
        )
        H[update].append(updated_mRNA)

        # Generating miRNA features
        print("Synthesizing miRNA features...")
        updated_miRNA = updateFeatures(
            mRNA, adj_input, miRNA_input, miRNA, hypers_miRNA, device, update, labels, dataset_name
        )
        H[update].append(updated_miRNA)
    return H[total_update - 1]

    # print(best, file=open(output_file, "a"))
    # print(f"Best auc score: {best} on the epoch {best_epoch}/{num_epochs}")
    # dd.to_csv(filename)
