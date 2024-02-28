from sklearn.metrics import roc_auc_score
import pandas as pd


def compute_metric(labels, pred):
    score = roc_auc_score(labels, pred)
    return score


LABELS_PATH = "./data/private_info/private.csv"  # либо "./data/private_info/public.csv"
SUBM_PATH = "./data/submission.csv"

if __name__ == "__main__":
    subm_df = pd.read_csv(SUBM_PATH, sep="\t")
    labels_df = pd.read_csv(LABELS_PATH, sep="\t")

    metric = compute_metric(labels_df, subm_df)
    print(f"ROC AUC: {metric}")
