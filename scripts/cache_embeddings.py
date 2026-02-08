import torch
import tqdm
import os
import logging
from src.utils import setup_logging
from src.data_loader import get_ood_dataloader
from src.feature_extractor import TiRexEmbedding
from src.config import REPO_CHRONOS, REPO_GIFTEVAL_PRETRAIN, CHRONOS_TRAIN, GIFTEVAL_TRAIN

setup_logging(log_name="cache_embeddings")
logger = logging.getLogger(__name__)

def cache_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)
    
    embedder = TiRexEmbedding(device=device).eval()
    
    training_repo_map = {
        REPO_CHRONOS: CHRONOS_TRAIN,
        # chronos extra training dataset would require extra handling: "Dataset scripts are no longer supported, but found chronos_datasets_extra.py"
        #REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA,
        REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN
    }
    raw_cache = {}

    for repo, configs in training_repo_map.items():
        for config in configs:
            logger.info(f"Caching Dataset: {config}")
            try:
                # batch size one should guarantee to get different time series
                loader = get_ood_dataloader({repo: [config]}, "train", batch_size=1)
                
                subset_embs = []
                pbar = tqdm.tqdm(total=1000, desc=f"Mining {config}")
                
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        if i >= 1000: 
                            break
                        
                        # raw extraction without normalization
                        raw = embedder.extract_raw(batch["inputs"].to(device))
                        
                        subset_embs.append(raw.cpu().to(torch.bfloat16))
                        pbar.update(1)
                
                pbar.close()
                if subset_embs:
                    raw_cache[config] = torch.cat(subset_embs, dim=0)
                    
            except Exception as e:
                logger.error(f"Error ocurred processing {config}: {e}")

    # save embeddings
    save_path = "outputs/raw_embeddings_cache.pt"
    torch.save(raw_cache, save_path)
    logger.info(f"Saved embeddings here: {save_path} ({len(raw_cache)} Subsets)")

if __name__ == "__main__":
    cache_embeddings()