import torch
import pandas as pd
import os
import logging
from tqdm import tqdm
from src.utils import setup_logging
from src.feature_extractor import TiRexEmbedding
from src.mahalanobis import Mahalanobis

setup_logging(log_name="calculate_scores")
logger = logging.getLogger(__name__)

def run_scoring():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs"
    cache_dir = "outputs/cache_parts"

    # loading baseline
    baseline_path = "outputs/baseline_stats.pt"
    if not os.path.exists(baseline_path):
        logger.error("Baseline file not found. Run compute_baseline.py first.")
        return
    
    all_baselines = torch.load(baseline_path, map_location=device)

    group_files = {
        "ID_TRAIN": "cache_id_train.pt",
        "ID_VAL": "cache_id_val.pt",
        "OOD_BENCHMARK": "cache_ood_benchmark.pt"
    }

    final_results = []

    for group_name, file_name in group_files.items():
        file_path = os.path.join(cache_dir, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"Cache file {file_name} missing.")
            continue
            
        logger.info(f"Scoring Group: {group_name}")
        group_cache = torch.load(file_path, map_location=device)

        for subset_name, content in group_cache.items():
            raw_embeddings = content['embeddings'].float()
            mses = content['mses'].numpy()
            mases = content['mases'].numpy()
            n_samples = len(raw_embeddings)
            
            for mode_name, stats in all_baselines.items():
                # initialize Mahalanobis class
                detector = Mahalanobis()
                detector.inv_covariance_matrix = stats['inv_cov']
                detector.means_tensor = stats['means_tensor']
                
                # apply the specific normalization for this mode
                use_l2 = "l2" in mode_name
                use_ln = "ln" in mode_name
                norm_embs = TiRexEmbedding.apply_normalization(raw_embeddings, use_l2=use_l2, use_ln=use_ln)
                
                # calculate OOD scores
                scores = detector.get_score(norm_embs).numpy()
                
                logger.debug(f"Computed {mode_name} scores for {subset_name}")

                for i in range(n_samples):
                    final_results.append({
                        "group": group_name,
                        "subset": subset_name,
                        "sample_idx": i,
                        "mse": mses[i],
                        "mase": mases[i],
                        "ood_score": scores[i],
                        "norm_mode": mode_name
                    })

    # save to a single CSV for easy plotting
    df = pd.DataFrame(final_results)
    save_path = os.path.join(output_dir, "final_ood_scores.csv")
    df.to_csv(save_path, index=False)
    
    logger.info(f"Score file saved to {save_path}")
    
    # a quick sanity check
    summary = df.groupby(['group', 'norm_mode'])['ood_score'].mean().unstack()
    print("Mean OOD scores")
    print(summary)

if __name__ == "__main__":
    run_scoring()