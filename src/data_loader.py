import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from itertools import chain
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
        for key in ['target', 'consumption_kW', 'values', 'price_mean', 'item_data', 'power_mw', 'state', 't_mean', 'temperature', 'generation_biomass']:
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
                if "Salesforce" in repo:
                    ds = load_dataset(repo, data_files={"train": f"{config}/**/*.arrow"}, split="train", streaming=True)
                elif "chronos_datasets_extra" in repo:
                    ds = load_dataset(repo, name=config, split="train", streaming=True, revision="refs/pr/1")
                else:
                    ds = load_dataset(repo, data_dir=config, split="train", streaming=True)
                
                it = iter(ds)
                try:
                    # peek at the first sample's length
                    first_example = next(it)
                    first_data = self._extract_actual_data(first_example)
                    first_len = len(first_data) if first_data is not None else 0
                except StopIteration: continue

                # strategy is based on the length of the first sequence
                if first_len > 100000:
                    # extremely long series -> draw all needed windows from it
                    skip_factor, samples_per_series = 1, self.total_samples_needed
                    logger.info(f"[{config}] Multi-Crop Logic (Len: {first_len})")
                else:
                    # case: standard series -> stride through the stream for diversity
                    skip_factor, samples_per_series = 10, 1
                    logger.info(f"[{config}] Strided Logic (Len: {first_len})")

                # chain back the first example and run the sampling loop
                stream = chain([first_example], it)
                samples_yielded = 0
                for i, example in enumerate(stream):
                    if samples_yielded >= self.total_samples_needed: break
                    if i % skip_factor != 0: continue 
                    
                    data = self._extract_actual_data(example)
                    if data is None: continue
                    
                    rem = self.total_samples_needed - samples_yielded
                    for inputs, targets in self._process_series(data, min(rem, samples_per_series)):
                        yield {
                            "inputs": inputs, "targets": targets, 
                            "is_val": (samples_yielded % val_step == 0), 
                            "subset": config
                        }
                        samples_yielded += 1
                        if samples_yielded >= self.total_samples_needed: break

def get_ood_dataloader(repo_map, split_mode="all", batch_size=32):
    dataset = TiRexStreamingDataset(repo_map=repo_map, split_mode=split_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)