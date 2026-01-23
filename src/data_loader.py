import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import logging
import numpy as np

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TiRexStreamingDataset(IterableDataset):
    def __init__(self, repo_map: dict, target_len: int = 512, stride: int = 256, split_mode: str = "train", val_ratio: float = 0.2, limit_configs: int = None):
        super().__init__()
        self.repo_map = repo_map
        self.target_len = target_len
        self.stride = stride
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
        priority_keys = ['target', 'consumption_kW', 'item_data', 'values']
        
        for key in priority_keys:
            if key in example and example[key] is not None:
                data = example[key]
                if isinstance(data, (list, np.ndarray, torch.Tensor)):
                    return data
        return None


    def _process_series(self, series):
        try:
            ts = torch.tensor(series, dtype=torch.float32)
        except Exception:
            return 
        # handle NaNs   
        ts = torch.nan_to_num(ts, nan=0.0)
        
        # ensure shape [Variates, TimeSteps]
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        elif ts.ndim > 2:
            # in case of more dimension e.g. batch dimension 
            ts = ts.view(-1, ts.shape[-1])

        # get the sequence length
        seq_len = ts.shape[-1]

        # generate 512-length segments
        segments = []

        # add padding if sequence is shorter than the specified target length
        if seq_len < self.target_len:
            pad_size = self.target_len - seq_len
            segments.append(F.pad(ts, (pad_size, 0), "constant", 0))
            
        # sliding window extraction to cover the entire time series
        else:
            for i in range(0, seq_len - self.target_len + 1, self.stride):
                segments.append(ts[:, i : i + self.target_len])
        
        # handles univariates [1, 512] and multivariates [x, 512]
        for seg in segments:
            for v in range(seg.shape[0]):
                # yields shape [1, 512] every time
                yield seg[v:v+1, :]


    def __iter__(self):
        val_step = int(1 / self.val_ratio) if self.val_ratio > 0 else 10**9
        
        for repo, config in self.all_configs:
            ds = None
            logger.info(f"Loading {repo} | subset: {config}")
            
            try:
                if "Salesforce" in repo:
                    # Salesforce data handling: recursive search in directory
                    ds = load_dataset(repo, data_files={"train": f"{config}/**/*.arrow"}, split="train", streaming=True)
                else:
                    # Chronos data handling
                    ds = load_dataset(repo, data_dir=config, split="train", streaming=True)
                
                if ds is None: continue

                for i, example in enumerate(ds):
                    # determine whether this is a val or train sample
                    is_val_sample = (i % val_step == 0)
                    if self.split_mode == "val" and not is_val_sample: continue
                    if self.split_mode == "train" and is_val_sample: continue
                        
                    # data extraction
                    data = self._extract_actual_data(example)
                    if data is None: continue
                    
                    # metadata for later OOD detection
                    metadata = {
                        "repo": repo,
                        "class": config,
                        "id": str(example.get('id', example.get('item_id', 'unknown')))
                    }
                    
                    for window in self._process_series(data):
                        yield {
                            "inputs": window,
                            "meta": metadata
                        }
            except Exception as e:
                logger.warning(f"Error in {repo}/{config}: {e}")
                continue


def get_ood_dataloader(repo_map, split_mode, batch_size=64, target_len=512, limit_configs=None):
    dataset = TiRexStreamingDataset(repo_map=repo_map, split_mode=split_mode, target_len=target_len, limit_configs=limit_configs)
    return DataLoader(dataset, batch_size=batch_size)