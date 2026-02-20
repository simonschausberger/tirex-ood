import torch
import os
import logging
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Kaggle setup for extra chronos datasets
os.environ["KAGGLE_CONFIG_DIR"] = "kaggle"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# setup logging
logger = logging.getLogger(__name__)

class TiRexStreamingDataset(IterableDataset):
    def __init__(self, repo_map, target_len=2048, prediction_len=8, total_samples_needed=1500, split_mode="all", val_ratio=0.2):
        super().__init__()
        self.repo_map = repo_map
        self.target_len = target_len
        self.prediction_len = prediction_len
        self.total_samples_needed = total_samples_needed
        self.split_mode = split_mode
        self.val_ratio = val_ratio

    def _extract_actual_data(self, example):
        # determine the data field based on common dataset schemas
        for key in ['target', 'consumption_kW', 'values', 'price_mean', 'item_data', 'power_mw', 'PRCP', 't_mean', 'temperature', 'generation_biomass', 'HUFL']:
            if key in example and example[key] is not None:
                return example[key]
        return None

    def _process_series(self, series, num_windows):
        # extraction logic: handles nan-padding for short or jump-sliding for long series
        try:
            ts = torch.tensor(series, dtype=torch.bfloat16)
            if ts.ndim == 1: ts = ts.unsqueeze(0)
            
            seq_len = ts.shape[-1]
            total_req = self.target_len + self.prediction_len

            if seq_len < total_req:
                pad_size = total_req - seq_len
                # mask with NaNs for TiRex/xLSTM compatibility
                padded = torch.full((ts.shape[0], total_req), float('nan'), dtype=torch.bfloat16)
                padded[:, pad_size:] = ts
                yield padded[:, :self.target_len], padded[:, self.target_len:]

            else:
                # jump-sliding for longer series
                stride = 16 
                max_start = seq_len - total_req
                
                possible_windows = (max_start // stride) + 1
                windows_to_take = min(num_windows, possible_windows)
                
                for w in range(windows_to_take):
                    start = w * stride
                    window = ts[:, start : start + total_req]
                    yield window[:, :self.target_len], window[:, self.target_len:]
        except Exception:
            return

    def __iter__(self):
        val_step = int(1 / self.val_ratio) if self.val_ratio > 0 else 5
        
        for repo, configs in self.repo_map.items():
            for config in configs:
                # dynamic dataset loading
                if "Salesforce" in repo:
                    ds = load_dataset(repo, data_files={"train": f"{config}/**/*.arrow"}, split="train", streaming=True)
                elif "chronos_datasets_extra" in repo:
                    ds = load_dataset(repo, name=config, split="train", streaming=True, trust_remote_code=True)
                else:
                    ds = load_dataset(repo, data_dir=config, split="train", streaming=True)
                
                samples_yielded = 0
                logger.info(f"--- Mining Subset: {config} ---")

                for example in ds:
                    if samples_yielded >= self.total_samples_needed: break
                    
                    data = self._extract_actual_data(example)
                    if data is None: continue
                    
                    # try to fill the entire remaining quota from the current series
                    rem = self.total_samples_needed - samples_yielded
                    
                    for inputs, targets in self._process_series(data, rem):
                        is_val = (samples_yielded % val_step == 0)
                        
                        # split filtering
                        if self.split_mode == "val" and not is_val: continue
                        if self.split_mode == "train" and is_val: continue

                        yield {
                            "inputs": inputs, "targets": targets, 
                            "is_val": is_val, "subset": config
                        }
                        
                        samples_yielded += 1
                        if samples_yielded >= self.total_samples_needed: break

def get_ood_dataloader(repo_map, split_mode="all", batch_size=32):
    dataset = TiRexStreamingDataset(repo_map=repo_map, split_mode=split_mode)
    return DataLoader(dataset, batch_size=batch_size)