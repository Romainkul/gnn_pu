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

    def __call__(self, features, pos_indices):
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
        reliable_negatives = (F < self.threshold)
        return reliable_negatives

class SpyMethod:
    def __init__(self, spy_ratio=0.1):
        self.spy_ratio = spy_ratio

    def __call__(self, features, labels, treshold_negatives=0.5, treshold_positives=None):
        # Split positives into spy and main positive sets
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        num_spies = int(len(pos_indices) * self.spy_ratio)
        spy_indices = pos_indices[:num_spies]
        
        # Mark spies as unlabeled
        labels[spy_indices] = -1

        # Train a classifier on remaining positives and unlabeled data
        clf = RandomForestClassifier()
        clf.fit(features.detach().cpu().numpy(), labels.detach().cpu().numpy() == 1)

        # Predict on all unlabeled samples to find reliable negatives
        probs = torch.tensor(clf.predict_proba(features.detach().cpu().numpy())[:, 1], device=features.device)  # Probabilities for positive class
        reliable_negatives = (probs < treshold_negatives)  # Threshold for negatives
        
        if treshold_positives is not None:
            reliable_positive_mask = probs > treshold_positives
            return reliable_negatives, reliable_positive_mask
        else:
            return reliable_negatives
class RocchioMethod:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, features, pos_indices, positive=False, positive_treshold=0.8):
        # Calculate the positive centroid
        pos_centroid = features[pos_indices].mean(dim=0)
        pos_centroid = F.normalize(pos_centroid, p=2, dim=0)

        # Calculate cosine similarity with all instances
        similarities = F.cosine_similarity(features, pos_centroid.unsqueeze(0), dim=1)
        
        reliable_negatives_mask = similarities < self.threshold
        if positive:
            reliable_positive_mask = similarities > positive_treshold
            return reliable_negatives_mask, reliable_positive_mask
        else:
            return reliable_negatives_mask        

class OneDNFMethod:
    def __init__(self, salient_feature_threshold=0.5):
        self.salient_feature_threshold = salient_feature_threshold

    def __call__(self, features, pos_indices):
        # Calculate feature saliency by averaging positive features
        pos_features = features[pos_indices]
        salient_features = (pos_features.mean(dim=0) > self.salient_feature_threshold).float()

        # Identify instances with no salient features
        mask = (features * salient_features).sum(dim=1) == 0
        reliable_negatives = mask
        return reliable_negatives

class KMeansMethod:
    def __init__(self, num_clusters=2):
        self.num_clusters = num_clusters

    def __call__(self, features, pos_indices, positive=True):
        # Ensure pos_indices is on the same device as features
        pos_indices = pos_indices.to(features.device)

        # Run K-means clustering
        kmeans = KMeans(n_clusters=self.num_clusters)
        clusters = torch.tensor(kmeans.fit_predict(features.detach().cpu().numpy()), device=features.device)

        # Identify clusters that contain positive samples
        pos_clusters = torch.unique(clusters[pos_indices])  # Get unique positive clusters
        is_positive_cluster = torch.isin(clusters, pos_clusters)  # Boolean mask for positive clusters

        # Masks for reliable negatives and positives
        reliable_negatives_mask = ~is_positive_cluster
        if positive:
            reliable_positives_mask = is_positive_cluster
            return reliable_negatives_mask, reliable_positives_mask
        else:
            return reliable_negatives_mask
class GenerativePUMethod:
    def __init__(self, num_components=1, threshold=0.5):
        self.num_components = num_components
        self.threshold = threshold

    def __call__(self, features, pos_indices):
        # Train a Gaussian Mixture Model (GMM) on positive samples
        pos_features = features[pos_indices].detach().cpu().numpy()
        gmm = GaussianMixture(n_components=self.num_components)
        gmm.fit(pos_features)

        # Calculate likelihoods for all instances
        likelihoods = torch.tensor(gmm.score_samples(features.detach().cpu().numpy()), device=features.device)

        # Select instances with low likelihood as reliable negatives
        reliable_negatives = (likelihoods < self.threshold)
        return reliable_negatives