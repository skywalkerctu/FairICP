import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# Repo imports (match adult_FairICP.py behavior)
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

base_path = os.path.join(os.getcwd(), '../../data/')
sys.path.append(os.path.join(os.getcwd(), "../.."))
sys.path.append(os.path.join(os.getcwd(), "../../others/continuous-fairness-master"))

import get_dataset
from FairICP import utility_functions
from FairICP import FairICP_learning

# rpy2 / KPC (assumes your environment is already working)
os.environ['R_HOME'] = "" # your path
os.environ['R_USER'] = "" # your path
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector

KPC = importr('KPC')
stat = KPC.KPCgraph


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kpc_stat_only(Yhat_prob: np.ndarray, Y: np.ndarray, A: np.ndarray, knn: int = 1) -> float:
    """
    KPCgraph statistic only (no permutation p-value).
    Mirrors the single-call part of adult_FairICP.py.
    """
    rYhat = robjects.r.matrix(FloatVector(Yhat_prob.T.flatten()),
                              nrow=Yhat_prob.shape[0], ncol=Yhat_prob.shape[1])
    rZ = robjects.r.matrix(FloatVector(A.T.flatten()),
                           nrow=A.shape[0], ncol=A.shape[1])
    rY = robjects.r.matrix(FloatVector(Y.astype(float)),
                           nrow=A.shape[0], ncol=1)
    return float(stat(Y=rYhat, X=rY, Z=rZ, Knn=knn)[0])


def kpc_pvalue_mc(Yhat_prob: np.ndarray, Y: np.ndarray, A: np.ndarray, specified_At: np.ndarray,
                  knn: int = 1, n_perm: int = 100) -> tuple[float, float]:
    """
    Compute (statistic, Monte-Carlo p-value) using the same approach as adult_FairICP.py.
    specified_At: shape (n_perm, n, dA) or (n_perm+?, n, dA) depending on how constructed;
                  we assume indexing specified_At[j] gives (n, dA).
    """
    rYhat = robjects.r.matrix(FloatVector(Yhat_prob.T.flatten()),
                              nrow=Yhat_prob.shape[0], ncol=Yhat_prob.shape[1])
    rY = robjects.r.matrix(FloatVector(Y.astype(float)),
                           nrow=A.shape[0], ncol=1)

    rZ = robjects.r.matrix(FloatVector(A.T.flatten()),
                           nrow=A.shape[0], ncol=A.shape[1])
    res_ = float(stat(Y=rYhat, X=rY, Z=rZ, Knn=knn)[0])

    res_list = np.zeros(n_perm, dtype=float)
    for j in range(n_perm):
        At = specified_At[j]
        rZt = robjects.r.matrix(FloatVector(At.T.flatten()),
                                nrow=A.shape[0], ncol=A.shape[1])
        res_list[j] = float(stat(Y=rYhat, X=rY, Z=rZt, Knn=knn)[0])

    p_val = (1.0 / (n_perm + 1)) * (1 + np.sum(res_list >= res_))
    return res_, float(p_val)


def main():
    dataset = "adult"
    dim = 2
    fold_index = 0  # first KFold split

    # Match adult_FairICP.py val behavior: KFold(..., shuffle=True, random_state=123)
    X_, A_, Y_ = get_dataset.get_full_dataset(base_path, dataset, dim=dim)
    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    splits = list(kf.split(X_))
    train_indices, test_indices = splits[fold_index]

    # Match fold seed convention in inner_func_validate
    seed = 123 + fold_index
    set_seed(seed)

    X = X_[train_indices]
    A = A_[train_indices]
    Y = Y_[train_indices]

    X_test = X_[test_indices]
    A_test = A_[test_indices]
    Y_test = Y_[test_indices]

    input_data_train = np.concatenate((A, X), axis=1)
    input_data_test = np.concatenate((A_test, X_test), axis=1)

    # ----- Single configuration (keep this simple for the smoke test) -----
    batch_size = 512
    lr_loss = 1e-3
    lr_dis = 1e-4
    epochs = 60
    checkpoints = [20, 40, 60]

    # Adaptive-μ controller defaults (you can tune later)
    adaptive_cfg = dict(
        adaptive_mu=True,
        mu0=1.0,          # initial lambda = 0.5
        mu_min=0.0,
        mu_max=20.0,      # lambda max ~0.95
        eta_mu=0.3,
        ema_beta=0.05,
        kpc_M=4096,
        kpc_K=4,
        lambda_stab=0.5,
        kpc_knn=1,
        mu_update_freq=1,
    )

    # Loss/model
    cost_pred = nn.CrossEntropyLoss()
    model_type = "deep_model"

    # IMPORTANT: keep discriminator steps on so fairness can “turn on” when mu increases
    model = FairICP_learning.EquiClassLearner(
        lr_loss=lr_loss,
        lr_dis=lr_dis,
        epochs=epochs,
        loss_steps=1,
        dis_steps=1,  # do NOT gate on a fixed mu here
        cost_pred=cost_pred,
        in_shape=X.shape[1],
        batch_size=batch_size,
        model_type=model_type,
        lambda_vec=0.0,    # ignored if adaptive_mu=True in your patched learner
        num_classes=2,
        A_shape=A.shape[1],
        **adaptive_cfg
    )

    # Train
    out_dir = f"./adult_adaptive_mu_debug_fold{fold_index}/"
    os.makedirs(out_dir, exist_ok=True)

    model.fit(input_data_train, Y, epochs_list=checkpoints)

    # Save mu history if your patched learner records it
    if hasattr(model, "mu_history") and len(getattr(model, "mu_history")) > 0:
        pd.DataFrame(model.mu_history).to_csv(os.path.join(out_dir, "mu_history.csv"), index=False)

    # Prepare permutations for p-value evaluation (same idea as adult_FairICP.py)
    specified_density_te = utility_functions.Class_density_estimation(Y_test, A_test, Y_test, A_test)
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50, 100, specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]  # indexing specified_At[j] -> (n_test, dA)

    rows = []
    for i, cp in enumerate(model.checkpoint_list):
        # Load checkpoint model/dis
        model.model = model.cp_model_list[i]
        model.dis = model.cp_dis_list[i]

        Yhat_out_test = model.predict(input_data_test)

        # utility loss as in adult_FairICP.py
        mis_model = 1 - utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)

        # KPC stat + p-value (MC)
        kpc_stat, kpc_p = kpc_pvalue_mc(Yhat_out_test, Y_test, A_test, specified_At, knn=1, n_perm=100)

        # Optional: DEO is expensive; skip for smoke test
        rows.append({
            "fold": fold_index,
            "seed": seed,
            "epoch": cp,
            "batch_size": batch_size,
            "lr_loss": lr_loss,
            "lr_dis": lr_dis,
            "loss": float(mis_model),
            "kpcg_nn": float(kpc_stat),
            "pval_kpcg_nn": float(kpc_p),
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
