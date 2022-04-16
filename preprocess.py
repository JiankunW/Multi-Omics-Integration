from typing import List, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


def data_cleaning(
    mRNA_file: str, miRNA_file: str, adj_file: str, label_file
) -> Mapping[str, np.ndarray]:

    mRNA = pd.read_csv(mRNA_file, delimiter=",", index_col=0)
    miRNA = pd.read_csv(miRNA_file, delimiter=",", index_col=0)
    adj = pd.read_csv(adj_file, index_col=0)
    labels = pd.read_csv(label_file, delimiter=",", index_col=0)

    # clear redundant data
    _, x_ind, y_ind = np.intersect1d(mRNA.columns, miRNA.columns, return_indices=True)
    _, x_ind1, y_ind1 = np.intersect1d(miRNA.index, adj.columns, return_indices=True)
    _, x_ind2, y_ind2 = np.intersect1d(mRNA.index, adj.index, return_indices=True)
    mRNA = mRNA.iloc[x_ind2, x_ind]
    miRNA = miRNA.iloc[x_ind1, y_ind]
    adj = adj.iloc[y_ind2, y_ind1]
    _, x_ind3, y_ind3 = np.intersect1d(miRNA.columns, labels.index, return_indices=True)
    mRNA = mRNA.iloc[:, x_ind3]
    miRNA = miRNA.iloc[:, x_ind3]
    labels = labels.iloc[y_ind3, :]

    mRNA = mRNA.fillna(0)
    miRNA = miRNA.fillna(0)
    adj[adj == 1] = -1
    adj[adj == 0] = 1

    mRNA_names = np.array(mRNA.index, dtype="U")
    miRNA_names = np.array(miRNA.index, dtype="U")
    sample_names = np.array(miRNA.columns, dtype="U")
    mRNA = np.array(mRNA, dtype="float32").transpose()  # nxp
    miRNA = np.array(miRNA, dtype="float32").transpose()  # nxm
    adj = np.array(adj, dtype="float32")  # pxm
    labels = np.array(labels, dtype="float32")  # nx(classification dimensions)

    data: Mapping[str, np.ndarray] = {
        "mRNA_names": mRNA_names,
        "miRNA_names": miRNA_names,
        "sample_names": sample_names,
        "mRNA_data": mRNA,
        "miRNA_data": miRNA,
        "adj_matrix": adj,
        "labels": labels,
    }

    return data


def optimize_data(
    data: Mapping[str, np.ndarray], num_feats: List
) -> Mapping[str, np.ndarray]:
    """optimized original data features: normalization and feature selection

    Args:
        data (Mapping[str, np.ndarray]): original data
        num_feats (List): number of top features to select.
            The 1st entry is for mRNA, the 2nd miRNA.

    Returns:
        Mapping[str, np.ndarray]: optimized data in dictionary
    """
    np.seterr(divide="ignore", invalid="ignore")

    mRNA = data["mRNA_data"]
    miRNA = data["miRNA_data"]
    adj = data["adj_matrix"]
    labels = data["labels"]
    # normalizing adjacecy matrix
    C = np.sqrt(np.outer(np.sum(np.absolute(adj), 0), np.sum(np.absolute(adj), 1)))
    adj_new = np.divide(adj, C.transpose())

    # feature normalization
    mRNA = StandardScaler().fit_transform(mRNA)
    miRNA = StandardScaler().fit_transform(miRNA)

    # feature selection
    m_p = SelectKBest(f_classif, k="all").fit(mRNA, labels[:, 0]).pvalues_
    mi_p = SelectKBest(f_classif, k="all").fit(miRNA, labels[:, 0]).pvalues_
    keep_m_index = np.argsort(m_p)[0 : num_feats[0]]
    keep_mi_index = np.argsort(mi_p)[0 : num_feats[1]]

    print("Feature Selection...")
    print(f"mRNA: {num_feats[0]}/{mRNA.shape[1]} features are selected.")
    print(f"miRNA: {num_feats[1]}/{miRNA.shape[1]} features are selected.")

    data_optimized: Mapping[str, np.ndarray] = {
        "mRNA_names": data["mRNA_names"][keep_m_index],
        "miRNA_names": data["miRNA_names"][keep_mi_index],
        "sample_names": data["sample_names"],
        "mRNA_data": mRNA[:, keep_m_index],
        "miRNA_data": miRNA[:, keep_mi_index],
        "adj_matrix": adj_new[keep_m_index, :][:, keep_mi_index],
        "labels": labels,
    }
    return data_optimized


def significant_feats(X: np.ndarray, y: np.ndarray, threshold: float = 0.001) -> None:
    """calculate significant features by chi test

    Args:
        X (np.ndarray): n by k data matrix.
            n is # of samples, k is # of features.
        y (np.ndarray): labels
        threshold (float): threshold of significant test (default: .001)

    """
    X_p = SelectKBest(f_classif, k="all").fit(X, y[:, 0]).pvalues_
    index = np.where(X_p < threshold)[0]
    print(f"There are {len(index)}/{X.shape[1]} significant features.")
