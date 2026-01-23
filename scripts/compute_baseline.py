import torch
import tqdm
import os
import logging
from src.utils import setup_logging
from src.data_loader import get_ood_dataloader
from src.feature_extractor import TiRexEmbedding
from src.mahalanobis import Mahalanobis
from src.config import (
    REPO_CHRONOS, REPO_CHRONOS_EXTRA, REPO_GIFTEVAL_PRETRAIN,
    CHRONOS_TRAIN, CHRONOS_TRAIN_EXTRA, GIFTEVAL_TRAIN
)

# initialize logging
setup_logging(log_name="compute_baseline")
logger = logging.getLogger(__name__)

def compute_baseline():
    logger.info("Baseline process started...")
    # setup environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)
    logger.info(f"Execution device: {device}")

    # map repos to their specific configs
    training_repo_map = {
        REPO_CHRONOS: CHRONOS_TRAIN,
        # chronos extra training dataset would require extra handling: "Dataset scripts are no longer supported, but found chronos_datasets_extra.py"
        #REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA,
        REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN
    }

    try:
        # delete limit configs to use entire dataset
        train_loader = get_ood_dataloader(training_repo_map, "train", limit_configs=2)
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader: {e}", exc_info=True)
        return

    # initialize TiRex and the Mahalanobis engine
    logger.info("Initializing TiRex embedding extractor and Mahalanobis engine.")
    embedder = TiRexEmbedding(device=device).eval()
    detector = Mahalanobis()

    # feature extraction and online stats update
    logger.info("Starting statistics accumulation on training distribution...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            inputs = batch["inputs"].to(device)
            class_labels = batch["meta"]["class"]
            
            # extract features from last hidden layer
            embeddings = embedder(inputs) 
            
            num_variates = inputs.shape[1]
            # Repeat each label 'num_variates' times to match flattened embeddings
            expanded_labels = [label for label in class_labels for _ in range(num_variates)]

            # update the statistics
            detector.update(embeddings, expanded_labels)

            if i % 500 == 0:
                logger.info(f"Step {i}: Total samples processed: {detector.n_total}")

    # finalize covariance and inversion
    logger.info("Finalizing covariance matrix and computing inversion")
    detector.finalize()

    # save the baseline
    save_path = "outputs/baseline_stats.pt"
    try:
        torch.save({
            "feature_dim": detector.feature_dim,
            "n_total": detector.n_total,
            "inv_cov": detector.inv_covariance_matrix,
            "means_tensor": detector.means_tensor,
            "class_names": detector.class_labels_sorted
        }, save_path)
        logger.info(f"Baseline statistics successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error while saving baseline statistics: {e}")

if __name__ == "__main__":
    compute_baseline()