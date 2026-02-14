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

# setup logging
setup_logging(log_name="cache_all_groups_final")
logger = logging.getLogger(__name__)

def process_and_cache(loader, limit, device, embedder, desc):
    all_embs, all_mses = [], []
    pbar = tqdm.tqdm(total=limit, desc=desc, leave=False)
    
    for batch in loader:
        if len(all_embs) >= limit: break

        inputs = batch["inputs"].to(device) 
        # shape: [B, 1, 2048]

        targets = batch["targets"].to(device)
        
        with torch.no_grad():
            # batch extraction of raw embeddings
            raw_embs = embedder(inputs) 
            
            # batch forecasting for mse calculation
            pred_input = inputs.squeeze(1) if inputs.ndim == 3 else inputs
            out = embedder.model.forecast(pred_input, prediction_length=targets.shape[-1])
            forecasts = out[0] if isinstance(out, tuple) else out
            
            # process individual samples in the batch
            for i in range(inputs.size(0)):
                if len(all_embs) >= limit: break
                
                # calculate median and mse on CPU to save GPU memory
                f_i = forecasts[i].median(dim=0).values.cpu()
                t_i = targets[i].flatten().cpu()
                
                # handle potential length mismatches
                c_len = min(f_i.size(0), t_i.size(0))
                if c_len == 0: continue
                
                mse = torch.mean((f_i[:c_len] - t_i[:c_len])**2).item()
                
                # convert to bfloat16 for storage efficiency
                all_embs.append(raw_embs[i].cpu().to(torch.bfloat16).unsqueeze(0))
                all_mses.append(mse)
                pbar.update(1)
    
    pbar.close()
    if not all_embs: return None
    return {"embeddings": torch.cat(all_embs, dim=0), "mses": torch.tensor(all_mses, dtype=torch.float32)}

def cache_all_groups():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs/cache_parts"
    os.makedirs(output_dir, exist_ok=True)
    
    embedder = TiRexEmbedding(device=device).eval()
    
    # mining parameters
    TRAIN_LIMIT, VAL_LIMIT, OOD_LIMIT, BATCH_SIZE = 1000, 200, 1000, 32
    id_map = {REPO_CHRONOS: CHRONOS_TRAIN, REPO_CHRONOS_EXTRA: CHRONOS_TRAIN_EXTRA, REPO_GIFTEVAL_PRETRAIN: GIFTEVAL_TRAIN}
    ood_map = {REPO_CHRONOS: CHRONOS_ZS_BENCHMARK, REPO_CHRONOS_EXTRA: CHRONOS_ZS_BENCHMARK_EXTRA, REPO_GIFTEVAL: GIFTEVAL_ZS_BENCHMARK}

    # cache paths
    train_path = os.path.join(output_dir, "cache_id_train.pt")
    val_path = os.path.join(output_dir, "cache_id_val.pt")
    ood_path = os.path.join(output_dir, "cache_ood_benchmark.pt")

    # load or init caches
    train_data = torch.load(train_path, map_location="cpu") if os.path.exists(train_path) else {}
    val_data = torch.load(val_path, map_location="cpu") if os.path.exists(val_path) else {}
    ood_data = torch.load(ood_path, map_location="cpu") if os.path.exists(ood_path) else {}

    # ID data (Train & Val)
    logger.info("ID Data")
    for repo, subsets in id_map.items():
        for subset in subsets:
            if subset in train_data and subset in val_data:
                logger.info(f"Skipping {subset} (cached)")
                continue

            # load full subse
            loader = get_ood_dataloader({repo: [subset]}, batch_size=BATCH_SIZE)
            
            # temporary storage for this subset's extraction
            tr_embs, tr_mses, vl_embs, vl_mses = [], [], [], []
            
            for batch in tqdm.tqdm(loader, desc=f"ID: {subset}"):
                if len(tr_embs) >= TRAIN_LIMIT and len(vl_embs) >= VAL_LIMIT: break
                
                # logic to split incoming batch into Train/Val arrays
                inputs, targets = batch["inputs"].to(device), batch["targets"].to(device)
                with torch.no_grad():
                    raw_embs = embedder(inputs)
                    out = embedder.model.forecast(inputs.squeeze(1), prediction_length=targets.shape[-1])
                    forecasts = out[0] if isinstance(out, tuple) else out
                    
                    for i in range(inputs.size(0)):
                        is_val = batch["is_val"][i].item()
                        if is_val and len(vl_embs) >= VAL_LIMIT: continue
                        if not is_val and len(tr_embs) >= TRAIN_LIMIT: continue
                        
                        # mse and CPU transfer
                        f_i = forecasts[i].median(dim=0).values.cpu()
                        t_i = targets[i].flatten().cpu()
                        mse = torch.mean((f_i[:t_i.size(0)] - t_i[:f_i.size(0)])**2).item()
                        emb = raw_embs[i].cpu().to(torch.bfloat16).unsqueeze(0)

                        if is_val:
                            vl_embs.append(emb); vl_mses.append(mse)
                        else:
                            tr_embs.append(emb); tr_mses.append(mse)

            if tr_embs:
                train_data[subset] = {"embeddings": torch.cat(tr_embs, dim=0), "mses": torch.tensor(tr_mses)}
                torch.save(train_data, train_path)
            if vl_embs:
                val_data[subset] = {"embeddings": torch.cat(vl_embs, dim=0), "mses": torch.tensor(vl_mses)}
                torch.save(val_data, val_path)

    # OOD data
    logger.info("Mining OOD Data")
    for repo, subsets in ood_map.items():
        for subset in subsets:
            if subset in ood_data: continue
            
            loader = get_ood_dataloader({repo: [subset]}, batch_size=BATCH_SIZE)
            res = process_and_cache(loader, OOD_LIMIT, device, embedder, f"OOD: {subset}")
            
            if res:
                ood_data[subset] = res
                torch.save(ood_data, ood_path)

    logger.info("All groups successfully cached.")

if __name__ == "__main__":
    cache_all_groups()