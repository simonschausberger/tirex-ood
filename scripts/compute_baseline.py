import torch
import os
import logging
from src.utils import setup_logging
from src.feature_extractor import TiRexEmbedding
from src.mahalanobis import Mahalanobis

# setup logging
setup_logging(log_name="compute_baseline")
logger = logging.getLogger(__name__)

def compute_baseline():
    logger.info("Baseline process started from cache...")
    output_dir = "outputs"
    cache_path = os.path.join(output_dir, "cache_parts", "cache_id_train.pt")
    
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found at {cache_path}!")
        return

    # load the raw embeddings
    logger.info(f"Loading cached embeddings from {cache_path}...")
    group_data = torch.load(cache_path, map_location="cpu")

    # extract all embeddings
    raw_cache = {name: content['embeddings'] for name, content in group_data.items()}
    logger.info(f"Using {len(raw_cache)} subsets for baseline calculation.")

    # normalization configurations
    norm_configs = [
        {"use_l2": True,  "use_ln": True,  "name": "l2_ln"},
        {"use_l2": False, "use_ln": False, "name": "raw"},
        {"use_l2": True,  "use_ln": False, "name": "only_l2"},
        {"use_l2": False, "use_ln": True,  "name": "only_ln"}
    ]
    
    # hold the final results for all modes
    multi_mode_results = {}

    for cfg in norm_configs:
        logger.info(f"Processing mode: {cfg['name']}")
        
        # apply static normalization
        norm_data = {
            name: TiRexEmbedding.apply_normalization(embs.float(), cfg['use_l2'], cfg['use_ln'])
            for name, embs in raw_cache.items()
        }
        
        # initialize detector and compute parameters
        detector = Mahalanobis()
        detector.compute_from_cache(norm_data)
        
        # store result
        multi_mode_results[cfg['name']] = {
            "feature_dim": detector.feature_dim,
            "n_total": detector.n_total,
            "inv_cov": detector.inv_covariance_matrix,
            "means_tensor": detector.means_tensor,
            "class_names": detector.class_labels_sorted
        }

    # save baselines
    save_path = "outputs/baseline_stats.pt"
    try:
        torch.save(multi_mode_results, save_path)
        logger.info(f"All baseline modes successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error while saving baseline statistics: {e}")

if __name__ == "__main__":
    compute_baseline()