import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def SVM(X_train, y_train, X_test, y_test) -> float:

    clf = svm.SVC(kernel="linear", probability=True).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]

    y_pred_bin = np.copy(y_pred)
    y_pred_bin[y_pred_bin < 0.5] = 0
    y_pred_bin[y_pred_bin >= 0.5] = 1

    return roc_auc_score(y_test, y_pred)


def SVM_multiclass(X_train, y_train, X_test, y_test) -> float:

    clf = svm.SVC(kernel="linear", probability=True).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    return roc_auc_score(y_test, y_pred, multi_class="ovo")


def evaluate(data_matrix, labels) -> float:
    """evaluate data matrix. output auc score.

    Args:
        data_matrix (np.ndarray): n by k feature matrix.
            n is # of samples, k # of features
        labels (np.ndarray): labels corresponding to each sample

    Returns:
        AUC score
    """
    trial = 50
    AUC_all = []
    dim_labels = labels.shape[1]

    for col in range(dim_labels):
        y = labels[:, col]
        cls_counts = len(np.unique(y))
        AUC = []
        for i in range(trial):
            X_train, X_test, y_train, y_test = train_test_split(
                data_matrix, y, test_size=0.2, random_state=i
            )
            if cls_counts == 2:
                auc = SVM(X_train, y_train, X_test, y_test)
            elif cls_counts > 2:
                auc = SVM_multiclass(X_train, y_train, X_test, y_test)
            AUC.append(auc)
        AUC_all.append(sum(AUC) / trial)

    AUC_final = sum(AUC_all) / dim_labels

    return AUC_final
