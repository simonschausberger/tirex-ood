import torch
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mahalanobis:
    def __init__(self):
        self.feature_dim = 0
        self.inv_covariance_matrix = None
        self.means_tensor = None
        self.class_labels_sorted = None
        self.n_total = 0

    def compute_from_cache(self, raw_cache, shrinkage=1e-4):
        logger.info("Computing Mahalanobis parameters...")
        
        # determine dimensions
        self.class_labels_sorted = sorted(raw_cache.keys())
        first_key = self.class_labels_sorted[0]
        self.feature_dim = raw_cache[first_key].shape[-1]
        
        # determine class centers
        self.means_tensor = torch.stack([
            torch.mean(raw_cache[label].to(torch.float32), dim=0) 
            for label in self.class_labels_sorted
        ])

        # pooled covariance
        all_centered_embs = []
        self.n_total = 0
        
        for label in self.class_labels_sorted:
            # conversion
            embs = raw_cache[label].to(torch.float64)
            mu_c = torch.mean(embs, dim=0)
            
            centered = embs - mu_c 
            
            all_centered_embs.append(centered)
            self.n_total += embs.size(0)

        # center all samples
        X_centered = torch.cat(all_centered_embs, dim=0)
        
        # covariance calculation: 1/N * sum( (x-mu)(x-mu)^T )
        cov = torch.mm(X_centered.T, X_centered) / self.n_total
        
        # shrinkage for numerical stability
        eye = torch.eye(self.feature_dim, dtype=torch.float64)
        avg_variance = torch.mean(torch.diag(cov))
        cov = (1 - shrinkage) * cov + (shrinkage * avg_variance * eye)
        
        # inversion of covariance matrix
        self.inv_covariance_matrix = torch.inverse(cov).to(torch.float32)
        
        logger.info(f"Finalized statistics for {self.n_total} samples.")

    def get_score(self, embeddings):
        if self.inv_covariance_matrix is None:
            logger.error("Detector not finalized!")
            raise RuntimeError("Mahalanobis engine is not initialized.")

        embeddings = embeddings.detach().to("cpu", dtype=torch.float32)
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # distance calculation to all classes simultaneously
        # delta shape: (BatchSize, NumClasses, FeatureDim)
        delta = embeddings.unsqueeze(1) - self.means_tensor.unsqueeze(0)
        
        # compute Mahalanobis distance: (x-mu)^T * Sigma^-1 * (x-mu)
        # temp shape: (BatchSize, NumClasses, FeatureDim)
        temp = torch.matmul(delta, self.inv_covariance_matrix)
        
        # summing across features to get final distance per class
        # distances shape: (BatchSize, NumClasses)
        distances = torch.sum(temp * delta, dim=-1)

        # OOD Score is the negative distance to the nearest class
        min_distances, _ = torch.min(distances, dim=1)
        
        return -min_distances