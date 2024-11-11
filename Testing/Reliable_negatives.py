import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class PositiveEnlargement:
    def __init__(self, alpha=0.5, threshold=0.3, num_iterations=10):
        self.alpha = alpha
        self.threshold = threshold
        self.num_iterations = num_iterations

    def select_negatives(self, features, pos_indices):
        # Construct affinity matrix W using Gaussian kernel on features
        dist_matrix = torch.cdist(features, features, p=2)  # Euclidean distance
        W = torch.exp(-dist_matrix ** 2)

        # Normalize W to form transition matrix S
        D_inv = torch.diag(torch.pow(W.sum(1), -0.5))
        S = D_inv @ W @ D_inv

        # Initialize label matrix F with positive nodes set to 1
        F = torch.zeros((features.size(0),), device=features.device)
        F[pos_indices] = 1.0

        # Iteratively propagate labels
        for _ in range(self.num_iterations):
            F = self.alpha * S @ F + (1 - self.alpha) * F

        # Select reliable negatives below the threshold
        reliable_negatives = (F < self.threshold).nonzero().squeeze()
        return reliable_negatives

class SpyMethod:
    def __init__(self, spy_ratio=0.1):
        self.spy_ratio = spy_ratio

    def select_negatives(self, features, labels):
        # Split positives into spy and main positive sets
        pos_indices = (labels == 1).nonzero().squeeze()
        num_spies = int(len(pos_indices) * self.spy_ratio)
        spy_indices = pos_indices[:num_spies]
        
        # Mark spies as unlabeled
        labels[spy_indices] = -1

        # Train a classifier on remaining positives and unlabeled data
        clf = RandomForestClassifier()
        clf.fit(features, labels == 1)

        # Predict on all unlabeled samples to find reliable negatives
        probs = clf.predict_proba(features)[:, 1]  # Probabilities for positive class
        reliable_negatives = (probs < 0.5).nonzero()[0]  # Threshold for negatives
        return reliable_negatives

class RocchioMethod:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def select_negatives(self, features, pos_indices):
        # Calculate the positive centroid
        pos_centroid = features[pos_indices].mean(dim=0)
        pos_centroid = F.normalize(pos_centroid, p=2, dim=0)

        # Calculate cosine similarity with all instances
        similarities = F.cosine_similarity(features, pos_centroid.unsqueeze(0), dim=1)
        
        # Select instances with similarity below threshold
        reliable_negatives = (similarities < self.threshold).nonzero().squeeze()
        return reliable_negatives

class OneDNFMethod:
    def __init__(self, salient_feature_threshold=0.5):
        self.salient_feature_threshold = salient_feature_threshold

    def select_negatives(self, features, pos_indices):
        # Calculate feature saliency by averaging positive features
        pos_features = features[pos_indices]
        salient_features = (pos_features.mean(dim=0) > self.salient_feature_threshold).float()

        # Identify instances with no salient features
        mask = (features * salient_features).sum(dim=1) == 0
        reliable_negatives = mask.nonzero().squeeze()
        return reliable_negatives

class KMeansMethod:
    def __init__(self, num_clusters=2):
        self.num_clusters = num_clusters

    def select_negatives(self, features, pos_indices):
        # Run K-means clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        clusters = kmeans.fit_predict(features)

        # Identify clusters that contain no positive samples
        pos_clusters = set(clusters[pos_indices].tolist())
        reliable_negatives = [i for i, cluster in enumerate(clusters) if cluster not in pos_clusters]
        return torch.tensor(reliable_negatives)

class GenerativePUMethod:
    def __init__(self, num_components=1, threshold=0.5):
        self.num_components = num_components
        self.threshold = threshold

    def select_negatives(self, features, pos_indices):
        # Train a Gaussian Mixture Model (GMM) on positive samples
        pos_features = features[pos_indices].cpu().numpy()
        gmm = GaussianMixture(n_components=self.num_components)
        gmm.fit(pos_features)

        # Calculate likelihoods for all instances
        likelihoods = torch.tensor(gmm.score_samples(features.cpu().numpy()), device=features.device)

        # Select instances with low likelihood as reliable negatives
        reliable_negatives = (likelihoods < self.threshold).nonzero().squeeze()
        return reliable_negatives
