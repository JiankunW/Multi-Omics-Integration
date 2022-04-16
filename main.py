from argparse import ArgumentParser

import torch

from classifier import evaluate
from network import omicsGAN
from preprocess import data_cleaning, optimize_data, significant_feats


def main(
    total_update: int,
    dataset_name: str,
    m_select: int,
    mi_select: int,
    test_threshold: float,
    gpu_num: int,
) -> None:
    mRNA_file = "data/" + dataset_name + "/mRNA.csv"
    miRNA_file = "data/" + dataset_name + "/miRNA.csv"
    adj_file = "data/" + dataset_name + "/adjacency.csv"
    label_file = "data/" + dataset_name + "/label.csv"

    torch.manual_seed(111)
    device = torch.device(
        "cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu"
    )
    print(device)

    # INPUT
    data_original = data_cleaning(mRNA_file, miRNA_file, adj_file, label_file)

    # Optimized features
    num_feats = [m_select, mi_select]  # number of top features to select
    data = optimize_data(data_original, num_feats)

    # significant feature test (Table 4)
    print("Number of significant features...")
    significant_feats(data["mRNA_data"], data["labels"], test_threshold)
    significant_feats(data["miRNA_data"], data["labels"], test_threshold)

    # Evaluate single omic data
    auc_m = evaluate(data["mRNA_data"], data["labels"])
    auc_mi = evaluate(data["miRNA_data"], data["labels"])
    print(f"AUC score for original mRNA: {auc_m}")
    print(f"AUC score for original miRNA: {auc_mi}")

    # Feature integration
    integrated_mRNA, integrated_miRNA = omicsGAN(
        data["mRNA_data"],
        data["miRNA_data"],
        data["adj_matrix"],
        data["labels"],
        total_update,
        device,
        dataset_name,
    )

    # Evaluate integrated omics data
    auc_m = evaluate(integrated_mRNA, data["labels"])
    auc_mi = evaluate(integrated_miRNA, data["labels"])
    print(f"AUC score for synthetic mRNA: {auc_m}")
    print(f"AUC score for synthetic miRNA: {auc_mi}")

    # significant feature test (Table 4) after GANs
    print("Number of significant features...")
    significant_feats(integrated_mRNA, data["labels"], test_threshold)
    significant_feats(integrated_miRNA, data["labels"], test_threshold)

    # Save features

    # Evaluate and plot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-K", "--total-update", type=int, help="total rounds of update", default=5,
    )
    parser.add_argument(
        "-d", "--dataset-name", type=str, help="Name of the dataset", default="BRCA",
    )
    parser.add_argument(
        "-s1",
        "--m-select",
        type=int,
        help="# of selected features of mrna",
        default=2048,
    )
    parser.add_argument(
        "-s2",
        "--mi-select",
        type=int,
        help="# of selected features of mirna",
        default=256,
    )
    parser.add_argument(
        "-T",
        "--test-threshold",
        type=float,
        help="p-value threshold for significant features",
        default=0.001,
    )
    parser.add_argument(
        "-g", "--gpu-num", type=int, help="gpu index", default=0,
    )
    args = parser.parse_args()
    print(args)
    main(**vars(args))
