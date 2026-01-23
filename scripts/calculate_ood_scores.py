import torch
import tqdm
import pandas as pd
import os
import logging
from src.utils import setup_logging
from src.data_loader import get_ood_dataloader
from src.feature_extractor import TiRexEmbedding
from src.mahalanobis import Mahalanobis
from src.config import (
    REPO_CHRONOS, REPO_CHRONOS_EXTRA, REPO_GIFTEVAL_PRETRAIN, REPO_GIFTEVAL,
    CHRONOS_TRAIN, CHRONOS_TRAIN_EXTRA, GIFTEVAL_TRAIN,
    CHRONOS_ZS_BENCHMARK, CHRONOS_ZS_BENCHMARK_EXTRA, GIFTEVAL_ZS_BENCHMARK
)

setup_logging(log_name="calculate_scores")
logger = logging.getLogger(__name__)

def run_scoring():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading baseline
    baseline_path = "outputs/baseline_stats.pt"
    if not os.path.exists(baseline_path):
        logger.error("Baseline file not found. Run compute_baseline.py first.")
        return
    
    checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
    detector = Mahalanobis()
    
    detector.inv_covariance_matrix = checkpoint['inv_cov'].to(device).to(torch.float64)
    detector.means_tensor = checkpoint['means_tensor'].to(device).to(torch.float64)
    
    embedder = TiRexEmbedding(device=device).eval()
    

    # defining the three group maps
    groups = {
        "ID_TRAIN": {
            "map": {REPO_CHRONOS: CHRONOS_TRAIN, 
                    #REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA, 
                    REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN},
            "split": "train"
        },
        "ID_VAL": {
            "map": {REPO_CHRONOS: CHRONOS_TRAIN, 
                    #REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA, 
                    REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN},
            "split": "val"
        },
        "OOD_BENCHMARK": {
            "map": {REPO_CHRONOS: CHRONOS_ZS_BENCHMARK, 
                    #REPO_CHRONOS_EXTRA: CHRONOS_ZS_BENCHMARK_EXTRA, 
                    REPO_GIFTEVAL: GIFTEVAL_ZS_BENCHMARK},
            "split": "train"
        }
    }

    results = []

    # process each group
    for group_name, config in groups.items():
        logger.info(f"Scoring group: {group_name}...")
        
        # limit_configs=1 for testing
        loader = get_ood_dataloader(config["map"], split_mode=config["split"], batch_size=64, limit_configs=2)

        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=group_name):
                inputs = batch["inputs"].to(device)
                meta = batch["meta"]
                
                # feature extraction
                embeddings = embedder(inputs)
                
                # ensure embeddings match detector device and dtype
                embeddings = embeddings.to(device).to(torch.float64)
                
                # vectorized score calculation
                batch_scores = detector.get_score(embeddings)
                
                # map metadata to embeddings
                num_variates = inputs.shape[1]
                
                # enroll metadata
                for b_idx in range(inputs.shape[0]): 
                    for v_idx in range(num_variates):
                        global_idx = b_idx * num_variates + v_idx
                        results.append({
                            "group": group_name,
                            "repo": meta["repo"][b_idx],
                            "dataset": meta["class"][b_idx],
                            "sample_id": meta["id"][b_idx],
                            "variate_idx": v_idx,
                            "score": batch_scores[global_idx].item()
                        })

    # save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("outputs/all_ood_scores.csv", index=False)
    logger.info("Scores saved to outputs/all_ood_scores.csv")

if __name__ == "__main__":
    run_scoring()