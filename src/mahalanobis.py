import torch
import numpy as np
import logging

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mahalanobis:
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim

        # class specific stats 
        self.class_sums = {}
        self.class_counts = {}
        
        # global stats
        self.running_mean = torch.zeros(feature_dim, dtype=torch.float64)
        self.M2 = torch.zeros((feature_dim, feature_dim), dtype=torch.float64)
        self.n_total = 0
        
        # inverse shared covariance matrix
        self.inv_covariance_matrix = None

        # needed for vectorization (speed)
        self.means_tensor = None

        self.class_labels_sorted = None


    def update(self, embeddings, class_labels):
        embeddings = embeddings.detach().to(torch.float64)
        # handle entire batch at once
        batch_size = embeddings.size(0)

        old_n = self.n_total
        self.n_total += batch_size

        batch_mean = torch.mean(embeddings, dim=0)
        delta_batch = batch_mean - self.running_mean

        centered_batch = embeddings - batch_mean
        self.M2 += torch.mm(centered_batch.T, centered_batch)

        if old_n > 0:
            # update of M2 (Welford algorithm)
            self.M2 += (old_n * batch_size / self.n_total) * torch.outer(delta_batch, delta_batch)
        
        # update of global mean
        self.running_mean += (batch_size / self.n_total) * delta_batch

        # class-specific updates
        # group batch by labels
        unique_labels, labels_indices = np.unique(class_labels, return_inverse=True)
        labels_indices = torch.from_numpy(labels_indices)
        
        for i, label in enumerate(unique_labels):
            mask = (labels_indices == i)
            label_data = embeddings[mask]
            count = label_data.size(0)
            
            if label not in self.class_sums:
                self.class_sums[label] = torch.zeros(self.feature_dim, dtype=torch.float64)
                self.class_counts[label] = 0
            
            # update sum and counts
            self.class_sums[label] += torch.sum(label_data, dim=0)
            self.class_counts[label] += count
    

    def finalize(self, shrinkage=1e-4):
        if self.n_total < 2:
            logger.error("Insufficient data to finalize statistics.")
            raise ValueError(f"Insufficient data to compute covariance.")
        
        # covariance matrix
        cov = self.M2 / (self.n_total - 1)

        # apply shirnkage which pushes the matrix away from being singular
        eye = torch.eye(self.feature_dim, dtype=torch.float64)
        avg_variance = torch.mean(torch.diag(cov))
        cov = (1 - shrinkage) * cov + (shrinkage * avg_variance * eye)

        # compute inverse of covariance matrix for Mahalanobis distance
        self.inv_covariance_matrix = torch.inverse(cov)

        # create means tensor sorted by class labels
        self.class_labels_sorted = sorted(self.class_sums.keys())
        self.means_tensor = torch.stack([
            self.class_sums[l] / self.class_counts[l] for l in self.class_labels_sorted
        ])

    
    def get_score(self, embeddings):
        if self.inv_covariance_matrix is None:
            logger.error("Please call finalize() before calculating scores.")
            raise RuntimeError("Mahalanobis engine is not finalized. Call finalize() first.")

        embeddings = embeddings.detach().to(torch.float64)
        if embeddings.dim() == 1: 
            embeddings = embeddings.unsqueeze(0)
        
        # vectorized distance calculation to all classes simultaneously
        # delta shape: (BatchSize, NumClasses, FeatureDim)
        delta = embeddings.unsqueeze(1) - self.means_tensor.unsqueeze(0)
        
        # matrix multiplication over the feature dimension
        temp = torch.matmul(delta, self.inv_covariance_matrix)
        
        # element-wise product and sum yields the Mahalanobis distance
        # distances shape: (BatchSize, NumClasses)
        distances = torch.sum(temp * delta, dim=-1)

        # score is the negative distance to the nearest class
        min_distances, _ = torch.min(distances, dim=1)
        
        return -min_distances
        