import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class TiRexStreamingDataset(IterableDataset):
    def __init__(self, repo_map, target_len=2048, prediction_len=64, total_samples_needed=1200, split_mode="all", val_ratio=0.2):
        super().__init__()
        self.repo_map = repo_map
        self.target_len = target_len
        self.prediction_len = prediction_len
        self.total_samples_needed = total_samples_needed
        self.split_mode = split_mode
        self.val_ratio = val_ratio

    def _extract_actual_data(self, example):
        # determine the data field based on common dataset schemas
        for key in ['target', 'consumption_kW', 'values', 'price_mean', 'item_data']:
            if key in example and example[key] is not None:
                return example[key]
        return None

    def _process_series(self, series, num_windows):
        # extract several random windows from a single time series to capture variance
        try:
            ts = torch.tensor(series, dtype=torch.bfloat16)
           
            if ts.ndim == 1: ts = ts.unsqueeze(0)
            
            seq_len = ts.shape[-1]
            total_req = self.target_len + self.prediction_len

            if seq_len < total_req:
                pad_size = total_req - seq_len
                # use nan filled tensor as a mask
                padded = torch.full((ts.shape[0], total_req), float('nan'), dtype=torch.bfloat16)
                padded[:, pad_size:] = ts
                yield padded[:, :self.target_len], padded[:, self.target_len:]

            else:
                # draw multiple crops for long series
                max_start = seq_len - total_req
                for _ in range(num_windows):
                    start = np.random.randint(0, max_start + 1)
                    window = ts[:, start : start + total_req]
                    yield window[:, :self.target_len], window[:, self.target_len:]
        except Exception:
            return

    def __iter__(self):
        val_step = int(1 / self.val_ratio) if self.val_ratio > 0 else 5
        
        for repo, configs in self.repo_map.items():
            for config in configs:
                # load dataset in streaming mode
                ds = load_dataset(repo, data_dir=config, split="train", streaming=True)
                
                # retrieve total number of series from dataset metadata
                try:
                    total_rows = ds.info.splits['train'].num_examples
                except Exception:
                    total_rows = None

                # calculate sampling factors based on available metadata
                if total_rows and total_rows > self.total_samples_needed:
                    # massive dataset: use striding to cover the full distribution
                    skip_factor = max(1, total_rows // self.total_samples_needed)
                    samples_per_series = 1
                elif total_rows and total_rows > 0:
                    # sparse dataset: draw multiple windows per series to fill the quota
                    skip_factor = 1
                    samples_per_series = int(np.ceil(self.total_samples_needed / total_rows))
                else:
                    # fallback for unknown stream lengths: moderate skip and adaptive draw
                    skip_factor = 1
                    samples_per_series = 10 

                samples_yielded = 0
                for i, example in enumerate(ds):
                    if samples_yielded >= self.total_samples_needed: break
                    # skip samples based on skip_factor for global representation
                    if i % skip_factor != 0: continue 
                    
                    data = self._extract_actual_data(example)
                    if data is None: continue

                    # calculate remaining quota and current draw count
                    remaining = self.total_samples_needed - samples_yielded
                    # reduce draw count if stream proves to be long to ensure diversity
                    current_draw_limit = samples_per_series if (total_rows or i < 100) else 1
                    num_to_draw = max(1, min(remaining, current_draw_limit))

                    for inputs, targets in self._process_series(data, num_to_draw):
                        is_val = (samples_yielded % val_step == 0)
                        if self.split_mode == "val" and not is_val: continue
                        if self.split_mode == "train" and is_val: continue

                        yield {"inputs": inputs, "targets": targets, "is_val": is_val, "subset": config}
                        samples_yielded += 1
                        if samples_yielded >= self.total_samples_needed: break

def get_ood_dataloader(repo_map, split_mode="all", batch_size=32):
    dataset = TiRexStreamingDataset(repo_map=repo_map, split_mode=split_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)