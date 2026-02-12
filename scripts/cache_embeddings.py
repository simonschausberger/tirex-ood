import torch
import tqdm
import os
import logging
from src.utils import setup_logging
from src.data_loader import get_ood_dataloader
from src.feature_extractor import TiRexEmbedding
from src.config import (
    REPO_CHRONOS, REPO_GIFTEVAL_PRETRAIN, CHRONOS_TRAIN, GIFTEVAL_TRAIN,
    REPO_CHRONOS_EXTRA, CHRONOS_TRAIN_EXTRA, REPO_GIFTEVAL, 
    CHRONOS_ZS_BENCHMARK, GIFTEVAL_ZS_BENCHMARK, CHRONOS_ZS_BENCHMARK_EXTRA
)

# setup logging
setup_logging(log_name="cache_all_groups_final")
logger = logging.getLogger(__name__)

# silent noisy libraries
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def cache_all_groups():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs/cache_parts"
    os.makedirs(output_dir, exist_ok=True)
    embedder = TiRexEmbedding(device=device).eval()
    
    # Configuration
    TRAIN_LIMIT = 1000
    VAL_LIMIT = 200
    OOD_LIMIT = 1000
    TARGET_LEN = 2048 

    # Define maps for ID and OOD
    id_map = {REPO_CHRONOS: CHRONOS_TRAIN, REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA, REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN}
    ood_map = {REPO_CHRONOS: CHRONOS_ZS_BENCHMARK, REPO_CHRONOS_EXTRA: CHRONOS_ZS_BENCHMARK_EXTRA, REPO_GIFTEVAL: GIFTEVAL_ZS_BENCHMARK}

    # Cache paths
    train_path = os.path.join(output_dir, "cache_id_train.pt")
    val_path = os.path.join(output_dir, "cache_id_val.pt")
    ood_path = os.path.join(output_dir, "cache_ood_benchmark.pt")

    # Load existing or create new
    train_data = torch.load(train_path, map_location="cpu") if os.path.exists(train_path) else {}
    val_data = torch.load(val_path, map_location="cpu") if os.path.exists(val_path) else {}
    ood_data = torch.load(ood_path, map_location="cpu") if os.path.exists(ood_path) else {}

    # --- PART 1: ID DATA (Combined Pass) ---
    logger.info("--- Processing ID Data (Single Pass: 1000 Train / 200 Val) ---")
    for repo, subsets in id_map.items():
        for subset in subsets:
            # Only skip if both splits are already complete
            if subset in train_data and subset in val_data:
                logger.info(f"Skipping {subset} (already cached)")
                continue

            logger.info(f"Mining {subset} (Combined Pass)")
            loader = get_ood_dataloader({repo: [subset]}, split_mode="all", batch_size=1, target_len=TARGET_LEN)
            
            tr_embs, tr_mses = [], []
            vl_embs, vl_mses = [], []
            pbar = tqdm.tqdm(desc=f"Progress {subset}", leave=False)
            
            for batch in loader:
                # Stop if both quotas are met
                if len(tr_embs) >= TRAIN_LIMIT and len(vl_embs) >= VAL_LIMIT:
                    break

                is_val = batch["is_val"][0]
                
                # Check if we still need samples for this specific split
                if is_val and len(vl_embs) >= VAL_LIMIT: continue
                if not is_val and len(tr_embs) >= TRAIN_LIMIT: continue

                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                actual_horizon = targets.shape[-1]
                if actual_horizon == 0: continue

                with torch.no_grad():
                    # feature extraction
                    raw_emb = embedder(inputs)
                    
                    # forecasting
                    pred_input = inputs.squeeze(1) if inputs.ndim == 3 else inputs
                    out = embedder.model.forecast(pred_input, prediction_length=actual_horizon)
                    forecast_tensor = out[0] if isinstance(out, tuple) else out
                    
                    # median calculation
                    median_pred = forecast_tensor.median(dim=0).values if forecast_tensor.ndim == 3 else forecast_tensor
                    flat_pred = median_pred.flatten()[:actual_horizon]
                    
                    # calculate mse
                    mse = torch.mean((flat_pred - targets.flatten())**2).item()
                    emb_cpu = raw_emb.cpu().to(torch.bfloat16)

                    if is_val:
                        vl_embs.append(emb_cpu)
                        vl_mses.append(mse)
                    else:
                        tr_embs.append(emb_cpu)
                        tr_mses.append(mse)
                    
                    pbar.update(1)

            pbar.close()

            # Save both splits incrementally
            if tr_embs:
                train_data[subset] = {"embeddings": torch.cat(tr_embs, dim=0), "mses": torch.tensor(tr_mses, dtype=torch.float32)}
                torch.save(train_data, train_path)
            if vl_embs:
                val_data[subset] = {"embeddings": torch.cat(vl_embs, dim=0), "mses": torch.tensor(vl_mses, dtype=torch.float32)}
                torch.save(val_data, val_path)
            logger.info(f"Saved {subset}: Train({len(tr_embs)}) Val({len(vl_embs)})")

    # --- PART 2: OOD BENCHMARK ---
    logger.info("--- Processing OOD Benchmark (1000 samples) ---")
    for repo, subsets in ood_map.items():
        for subset in subsets:
            if subset in ood_data:
                logger.info(f"Skipping {subset} (already cached)")
                continue

            logger.info(f"Mining OOD {subset}")
            loader = get_ood_dataloader({repo: [subset]}, split_mode="train", batch_size=1, target_len=TARGET_LEN)
            
            embs, mses = [], []
            pbar = tqdm.tqdm(total=OOD_LIMIT, desc=f"Progress {subset}", leave=False)
            
            for batch in loader:
                if len(embs) >= OOD_LIMIT: break
                
                inputs, targets = batch["inputs"].to(device), batch["targets"].to(device)
                actual_horizon = targets.shape[-1]
                if actual_horizon == 0: continue

                with torch.no_grad():
                    raw_emb = embedder(inputs)
                    pred_input = inputs.squeeze(1) if inputs.ndim == 3 else inputs
                    out = embedder.model.forecast(pred_input, prediction_length=actual_horizon)
                    forecast_tensor = out[0] if isinstance(out, tuple) else out
                    median_pred = forecast_tensor.median(dim=0).values if forecast_tensor.ndim == 3 else forecast_tensor
                    mse = torch.mean((median_pred.flatten()[:actual_horizon] - targets.flatten())**2).item()
                    
                    embs.append(raw_emb.cpu().to(torch.bfloat16))
                    mses.append(mse)
                    pbar.update(1)

            pbar.close()
            if embs:
                ood_data[subset] = {"embeddings": torch.cat(embs, dim=0), "mses": torch.tensor(mses, dtype=torch.float32)}
                torch.save(ood_data, ood_path)

    logger.info(f"Success! All data cached.")

if __name__ == "__main__":
    cache_all_groups()