import numpy as np
import pandas as pd

# load score csv
df = pd.read_csv("outputs/final_ood_scores.csv")

mode_order = ["raw", "only_ln", "l2_ln", "only_l2"]
df["norm_mode"] = pd.Categorical(df["norm_mode"], categories=mode_order, ordered=True)

# compute FPR@95%TPR per OOD dataset
results = []

for norm_mode, df_m in df.groupby("norm_mode"):
     # compute the threshold using ID_train only
    id_scores = df_m[df_m["group"] == "ID_TRAIN"]["ood_score"].values
    if len(id_scores) == 0:
        continue

    # 95% TPR threshold
    threshold = np.percentile(id_scores, 5)

    print(f"Threshold @95%TPR: {threshold:.4f} | norm mode: {norm_mode}")

    for subset, group in df_m[df_m["group"] == "OOD_BENCHMARK"].groupby("subset"):
        ood_scores = group["ood_score"].values

        fpr = np.mean(ood_scores >= threshold)
        mean_score = np.mean(ood_scores)

        results.append({
            "norm_mode": norm_mode,
            "subset": subset,
            "FPR@95TPR": fpr,
            "mean_ood_score": mean_score,
            "num_samples": len(ood_scores)
        })

results_df = (
    pd.DataFrame(results)
    .sort_values(["norm_mode", "FPR@95TPR", "subset"])
    .reset_index(drop=True)
)
results_df.to_csv("outputs/fpr95tpr.csv", index=False)
