import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from omics1 import omics1
from omics2 import omics2


def main(
    total_update: int, dataset_name: str
) -> None:
    mRNA_file = "data/" + dataset_name + "/mRNA.csv"
    miRNA_file = "data/" + dataset_name + "/miRNA.csv"
    adj_file = "data/" + dataset_name + "/adjacency.csv"
    label_file = "data/" + dataset_name + "/label.csv"

    torch.manual_seed(111)
    gpu_num = 0  # int(sys.argv[2])
    device = torch.device(
        "cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu"
    )
    print(device)

    for i in range(1, total_update + 1):
        omics2(i, mRNA_file, miRNA_file, adj_file, label_file, device)
        omics1(i, mRNA_file, miRNA_file, adj_file, label_file, device)

    best_mRNA = pd.read_csv("best_mRNA.txt", header=None)
    keep_mRNA = np.argsort(best_mRNA.values, axis=0)[::-1][0][0]

    best_miRNA = pd.read_csv("best_miRNA.txt", header=None)
    keep_miRNA = np.argsort(best_miRNA.values, axis=0)[::-1][0][0]

    for i in range(1, total_update + 1):
        if i != keep_mRNA:
            os.remove("mRNA_BRCA" + str(i) + ".csv")

        if i != keep_miRNA:
            os.remove("miRNA_BRCA" + str(i) + ".csv")

    os.remove("best_mRNA.txt")
    os.remove("best_miRNA.txt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-K", "--total-update", type=int, help="total rounds of update", default=5,
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        help="Name of the dataset",
        default="stomach",
    )
    args = parser.parse_args()
    main(**vars(args))
