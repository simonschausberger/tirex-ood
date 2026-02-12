import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import logging
import numpy as np
import warnings

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mute noisy internal libraries to see actual errors
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class TiRexStreamingDataset(IterableDataset):
    def __init__(self, repo_map: dict, target_len: int = 2048, prediction_len: int = 64, num_samples_per_series: int = 1, split_mode: str = "train", val_ratio: float = 0.2, limit_configs: int = None):
        super().__init__()
        self.repo_map = repo_map
        self.target_len = target_len
        self.prediction_len = prediction_len
        self.num_samples_per_series = num_samples_per_series
        self.split_mode = split_mode
        self.val_ratio = val_ratio
        
        # collect the subsets
        self.all_configs = []
        for repo, allowed_configs in repo_map.items():
            try:
                i = 0
                for c in allowed_configs:
                    if limit_configs is not None and i >= limit_configs:
                        break
                    self.all_configs.append((repo, c))
                    i += 1
            except Exception as e:
                logger.error(f"Could not load configs for {repo}: {e}")

    def _extract_actual_data(self, example):
        # most common fields for time series data
        priority_keys = ['target', 'consumption_kW', 'item_data', 'values', 'price_mean']
        
        for key in priority_keys:
            if key in example and example[key] is not None:
                data = example[key]
                if isinstance(data, (list, np.ndarray, torch.Tensor)):
                    return data
        return None

    def _process_series(self, series):
        try:
            ts = torch.tensor(series, dtype=torch.bfloat16)
        
            # handle NaNs   
            ts = torch.nan_to_num(ts, nan=0.0)
            
            # ensure shape [Variates, TimeSteps]
            if ts.ndim == 1:
                ts = ts.unsqueeze(0)
            elif ts.ndim > 2:
                # in case of more dimension e.g. batch dimension 
                ts = ts.view(-1, ts.shape[-1])
                
            # multivariate handling
            if ts.shape[0] > 1:
                ts = ts[0:1, :] 

            # get the sequence length
            seq_len = ts.shape[-1]
            total_required = self.target_len + self.prediction_len

            # Handle sequences that are too short by zero-padding
            if seq_len < total_required:
                    pad_size = total_required - seq_len
                    # Pad at the beginning (left padding)
                    padded_ts = torch.zeros((ts.shape[0], total_required), dtype=torch.bfloat16)
                    padded_ts[:, pad_size:] = ts
                    
                    # split into inputs (context) and targets (horizon)
                    inputs = padded_ts[:, :self.target_len]
                    targets = padded_ts[:, self.target_len:]
                    yield inputs, targets
            else:
                    # random cropping applied to longer time series
                    max_start = seq_len - total_required
                    for _ in range(self.num_samples_per_series):
                        start = np.random.randint(0, max_start + 1)
                        full_window = ts[:, start : start + total_required]
                        
                        # split into inputs and targets
                        inputs = full_window[:, :self.target_len]
                        targets = full_window[:, self.target_len:]
                        yield inputs, targets

        except Exception as e:
            logger.debug(f"Error processing series: {e}")
            return

    def __iter__(self):
        val_step = int(1 / self.val_ratio) if self.val_ratio > 0 else 10**9
        
        for repo, config in self.all_configs:
            ds = None
            logger.info(f"Loading {repo} | subset: {config}")
            
            try:
                if "Salesforce" in repo:
                    ds = load_dataset(repo, data_files={"train": f"{config}/**/*.arrow"}, split="train", streaming=True)
                elif "chronos_datasets_extra" in repo:
                    ds = load_dataset(repo, name=config, split="train", streaming=True, revision="refs/pr/1")
                else:
                    ds = load_dataset(repo, data_dir=config, split="train", streaming=True)
                
                if ds is None: continue

                for i, example in enumerate(ds):
                    try:
                        is_val_sample = (i % val_step == 0)
                        
                        # split filtering logic
                        if self.split_mode != "all":
                            if self.split_mode == "val" and not is_val_sample: continue
                            if self.split_mode == "train" and is_val_sample: continue
                            
                        data = self._extract_actual_data(example)
                        if data is None: continue
                        
                        metadata = {
                            "repo": repo,
                            "class": config,
                            "id": str(example.get('id', example.get('item_id', 'unknown')))
                        }
                        
                        for inputs, targets in self._process_series(data):
                            yield {
                                "inputs": inputs, 
                                "targets": targets,
                                "is_val": is_val_sample,
                                "meta": metadata
                            }
                    except Exception as sample_err:
                        logger.debug(f"Skipping sample {i} in {config}: {sample_err}")
                        continue

            except Exception as e:
                logger.warning(f"Error in DataLoader when iterating over {repo}/{config}: {e}")
                continue


def get_ood_dataloader(repo_map, split_mode, batch_size=64, target_len=2048, prediction_len=64, limit_configs=None):
    dataset = TiRexStreamingDataset(
        repo_map=repo_map, 
        split_mode=split_mode, 
        target_len=target_len, 
        prediction_len=prediction_len, 
        limit_configs=limit_configs
    )
    return DataLoader(dataset, batch_size=batch_size)