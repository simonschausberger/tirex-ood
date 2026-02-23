import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# load score csv
df = pd.read_csv("outputs/final_ood_scores.csv")

mode_order = ["raw", "only_ln", "l2_ln", "only_l2"]
df["norm_mode"] = pd.Categorical(df["norm_mode"], categories=mode_order, ordered=True)

# compute metrics per OOD dataset
results = []

for norm_mode, df_m in df.groupby("norm_mode"):
     # compute the threshold using ID_train only
    id_scores = df_m[df_m["group"] == "ID_TRAIN"]["ood_score"].values
    if len(id_scores) == 0:
        continue

    # 95% TPR threshold
    threshold = np.percentile(id_scores, 5)

    for subset, group in df_m[df_m["group"] == "OOD_BENCHMARK"].groupby("subset"):
        ood_scores = group["ood_score"].values
        
        # FPR@95%TPR: fraction of OOD samples above ID threshold
        fpr = np.mean(ood_scores >= threshold)

        # AUROC calculation: y_true 0 for ID, 1 for OOD
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        
        # -ood_score as the probability of being OOD
        y_scores = np.concatenate([-id_scores, -ood_scores])
        auroc = roc_auc_score(y_true, y_scores)

        # mean ood score for subset
        mean_score = np.mean(ood_scores)

        results.append({
            "norm_mode": norm_mode,
            "subset": subset,
            "FPR@95TPR": fpr,
            "AUROC": auroc,
            "mean_ood_score": mean_score,
            "num_samples": len(ood_scores)
        })

results_df = (
    pd.DataFrame(results)
    .sort_values(["norm_mode", "AUROC", "subset"], ascending=[True, False, True])
    .reset_index(drop=True)
)
results_df.to_csv("outputs/metrics_summary.csv", index=False)
print("Metrics saved to outputs/metrics_summary.csv")
