import torch
import tqdm
import os
import logging
from src.utils import setup_logging
from src.feature_extractor import TiRexEmbedding
from src.data_loader import get_ood_dataloader
from src.config import (
    REPO_CHRONOS, REPO_GIFTEVAL_PRETRAIN, CHRONOS_TRAIN, GIFTEVAL_TRAIN,
    REPO_CHRONOS_EXTRA, CHRONOS_TRAIN_EXTRA, REPO_GIFTEVAL, 
    CHRONOS_ZS_BENCHMARK, GIFTEVAL_ZS_BENCHMARK, CHRONOS_ZS_BENCHMARK_EXTRA
)

# suppress noisy http and dataset logging
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)

# setup logging
setup_logging(log_name="cache_all_groups_final")
logger = logging.getLogger(__name__)

def cache_all_groups():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs/cache_parts"
    os.makedirs(output_dir, exist_ok=True)
    embedder = TiRexEmbedding(device=device).eval()
    
    # define dataset groupings for ID and OOD
    id_map = {REPO_CHRONOS: CHRONOS_TRAIN, REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA, REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN}
    ood_map = {REPO_CHRONOS: CHRONOS_ZS_BENCHMARK, REPO_CHRONOS_EXTRA: CHRONOS_ZS_BENCHMARK_EXTRA, REPO_GIFTEVAL: GIFTEVAL_ZS_BENCHMARK}

    # caching paths and initialization
    paths = {k: os.path.join(output_dir, f"cache_{k}.pt") for k in ["id_train", "id_val", "ood_benchmark"]}
    caches = {k: torch.load(paths[k], map_location="cpu") if os.path.exists(paths[k]) else {} for k in paths}

    # ID and OOD processing
    for group_name, current_map in [("ID", id_map), ("OOD", ood_map)]:
        logger.info(f"Processing {group_name} data")
        for repo, subsets in current_map.items():
            for subset in subsets:
                if group_name == "ID" and subset in caches["id_train"]:
                    logger.info(f"Skipping {subset} ({group_name}) - already in cache.")
                    continue
                if group_name == "OOD" and subset in caches["ood_benchmark"]:
                    logger.info(f"Skipping {subset} ({group_name}) - already in cache.")
                    continue

                loader = get_ood_dataloader({repo: [subset]}, batch_size=32)
                tr_embs, vl_embs = [], []
                
                # set total batches for progress bar visibility (1200 / 32 ~= 38)
                total_est = 38 if group_name == "ID" else 32
                pbar = tqdm.tqdm(loader, total=total_est, desc=f"{group_name}: {subset}", dynamic_ncols=True)
                
                for batch in pbar:
                    # break if limits are reached (1000 for Train/OOD, 200 for Val)
                    if group_name == "ID" and len(tr_embs) >= 1000 and len(vl_embs) >= 200: break
                    if group_name == "OOD" and len(tr_embs) >= 1000: break

                    inputs = batch["inputs"].to(device)
                    with torch.no_grad():
                        # store raw features
                        raw_embs = embedder(inputs).cpu().to(torch.bfloat16)
                        
                        for i in range(inputs.size(0)):
                            is_val = batch["is_val"][i].item()
                            emb = raw_embs[i].unsqueeze(0)
                            if group_name == "ID":
                                if is_val and len(vl_embs) < 200: vl_embs.append(emb)
                                elif not is_val and len(tr_embs) < 1000: tr_embs.append(emb)
                            else:
                                if len(tr_embs) < 1000: tr_embs.append(emb)
                    
                    # update progress bar with current counts
                    pbar.set_postfix({"tr": len(tr_embs), "vl": len(vl_embs)})

                pbar.close()

                if tr_embs:
                    key = "id_train" if group_name == "ID" else "ood_benchmark"
                    caches[key][subset] = {"embeddings": torch.cat(tr_embs, 0)}
                    torch.save(caches[key], paths[key])
                    # log the number of extracted embeddings for train/ood
                    logger.info(f"Extracted {len(tr_embs)} embeddings for {subset} ({key})")

                if vl_embs:
                    caches["id_val"][subset] = {"embeddings": torch.cat(vl_embs, 0)}
                    torch.save(caches["id_val"], paths["id_val"])
                    # log the number of extracted embeddings for validation
                    logger.info(f"Extracted {len(vl_embs)} validation embeddings for {subset}")

    logger.info("Caching complete.")

if __name__ == "__main__":
    cache_all_groups()