import logging
import math
import numbers
import numpy as np
from warnings import warn

from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin, clone
from sklearn.ensemble._bagging import BaseBagging
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    _num_features,
)
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBClassifier


##############################################################################
#                      WeightedIsoForest (Anomaly Detector)
##############################################################################
class WeightedIsoForest(OutlierMixin, BaseBagging):
    """
    A 'nearest-neighbors'-weighted Isolation Forest for anomaly detection, 
    adapted for scikit-learn >= 1.2.

    This class modifies the standard Isolation Forest so that points closer 
    to known positives (in a PU or semi-supervised setting) are more likely 
    to be flagged as anomalies among primarily negative data.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators (ExtraTreeRegressor) in the ensemble.
    type_weight : str, default='nearest_neighbors'
        Type of weighting scheme to apply. Currently, 'nearest_neighbors' 
        is supported, which weights path lengths by a node's local proportion 
        of known positives.
    n_neighbors : 'auto' or int, default='auto'
        Number of neighbors used if 'nearest_neighbors' weighting is chosen. 
        If 'auto', uses int(floor(sqrt(n_samples))).
    max_knn_sample : str or int, default='auto'
        Not used directly in the current implementation; placeholder for 
        future usage (subsampling KNN).
    max_samples : 'auto' or int or float, default='auto'
        Number of samples to draw to train each tree. If 'auto', uses 
        min(256, n_negatives). If int, directly use that number. If float, 
        use that fraction of n_negatives.
    contamination : 'auto' or float, default='auto'
        Not actively used. If 'auto', offset_ is set to -0.5 for consistency 
        with iForest scoring conventions. 
    max_features : float, default=1.0
        Fraction of features to draw from X to train each base estimator.
    bootstrap : bool, default=False
        Whether samples are drawn with replacement.
    n_jobs : int, default=-1
        Number of jobs for parallel processing. 
    behaviour : str, default='deprecated'
        Placeholder for older scikit-learn versions; ignored in recent versions.
    random_state : int or None, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Controls verbosity of the training process.
    warm_start : bool, default=False
        If True, reuse solution of previous calls to fit and add more estimators.
    pos_label : int or None, default=None
        Not used in this implementation; WeightedIsoForest takes a separate 
        pos_label array in its fit method.

    Attributes
    ----------
    nn_ : KNeighborsClassifier or None
        A k-NN classifier fit on the entire dataset (if type_weight='nearest_neighbors').
    nn_weight_ : ndarray of shape (n_samples,) or None
        The stored p(positive) for each sample if using nearest_neighbors weighting.
    depths_ : ndarray of shape (n_samples,)
        Stored sum of path lengths across all trees for each sample in score_samples().
    offset_ : float
        Offset used in decision_function() to shift the anomaly score threshold.

    References
    ----------
    Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation Forest."
    2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
    """

    def __init__(
        self,
        n_estimators=100,
        type_weight='nearest_neighbors',
        n_neighbors='auto',
        max_knn_sample="auto",
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        behaviour='deprecated',
        random_state=None,
        verbose=0,
        warm_start=False,
        pos_label=None
    ):
        super().__init__(
            estimator=ExtraTreeRegressor(
                max_features=1,   # override from max_features argument
                splitter='random',
                random_state=random_state
            ),
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

        self.behaviour = behaviour
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.max_knn_sample = max_knn_sample
        self.pos_label = pos_label
        self.type_weight = type_weight

        self.nn_ = None
        self.depths_ = None
        self.nn_weight_ = None

    def _set_oob_score(self, X, y):
        """Not implemented. Isolation Forest doesn't support OOB scores."""
        raise NotImplementedError("OOB score not supported by WeightedIsoForest.")

    def _get_estimator(self, random_state=None):
        """
        Required by scikit-learn (>= 1.2) to avoid abstract class errors.
        Must clone `self.estimator` rather than `base_estimator`.
        """
        estimator = clone(self.estimator)
        if random_state is not None:
            estimator.set_params(random_state=random_state)
        return estimator

    def fit(self, X, pos_label, y=None, sample_weight=None):
        """
        Fit the WeightedIsoForest on (assumed) negative data, i.e. rows where 
        pos_label == 0. If type_weight=='nearest_neighbors', also fit a KNN 
        to measure proximity to known positives.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        pos_label : array-like of shape (n_samples,)
            Binary labels {0,1}. 1 => known positive, 0 => unlabeled negative.
        y : Ignored
            Not used, present here for consistency with scikit-learn.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights if needed.
        """
        if issparse(X) or issparse(pos_label):
            logging.info("Converting sparse inputs to dense.")
        self.X = np.asarray(X) if not issparse(X) else X.toarray()
        self.pos_label = (
            pos_label.toarray() if issparse(pos_label) else np.asarray(pos_label)
        )

        # Check binary
        unique_vals = np.unique(self.pos_label)
        if not (len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals):
            raise ValueError("pos_label must contain exactly {0,1}.")

        # Subset negative data only
        X_neg = self.X[self.pos_label == 0]
        X_neg = check_array(X_neg, accept_sparse=False)
        rnd = check_random_state(self.random_state)

        # Record number of features
        self.n_features_in_ = X_neg.shape[1]

        # Resolve max_samples
        n_samples = X_neg.shape[0]
        if isinstance(self.max_samples, str):
            if self.max_samples == 'auto':
                max_samples_ = min(256, n_samples)
            else:
                raise ValueError(
                    f'max_samples={self.max_samples} not supported. '
                    'Use "auto", int, or float.'
                )
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    f"max_samples={self.max_samples} > n_samples={n_samples}. "
                    f"Will use n_samples instead."
                )
                max_samples_ = n_samples
            else:
                max_samples_ = self.max_samples
        else:  # float
            if not 0. < self.max_samples <= 1.:
                raise ValueError(f"max_samples={self.max_samples} must be in (0,1].")
            max_samples_ = int(self.max_samples * n_samples)

        self.max_samples_ = max_samples_
        max_depth = int(np.ceil(np.log2(max(max_samples_, 2))))

        # Fit ensemble on negative-only data
        super()._fit(
            X_neg,
            y=rnd.uniform(size=X_neg.shape[0]),  # random y to build random trees
            max_samples=max_samples_,
            max_depth=max_depth,
            sample_weight=sample_weight
        )

        # Fit k-NN if neighbor weighting is used
        if self.type_weight == 'nearest_neighbors':
            n_neighbors_ = self.n_neighbors
            if n_neighbors_ == 'auto':
                n_neighbors_ = int(np.floor(np.sqrt(len(self.pos_label))))
                n_neighbors_ = max(n_neighbors_, 1)

            self.nn_ = KNeighborsClassifier(
                n_neighbors=n_neighbors_,
                weights='uniform',
                algorithm='auto',
                n_jobs=-1
            )
            # Fit on all data (pos + neg) to measure distance to known positives
            self.nn_.fit(self.X, self.pos_label)

        # Set offset based on contamination
        if self.contamination == "auto":
            self.offset_ = -0.5
        else:
            self.offset_ = 0.0

        return self

    def predict(self, X):
        """
        Predict inlier (+1) or outlier (-1) for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            +1 for inliers, -1 for outliers.
        """
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, accept_sparse=False)
        scores = self.decision_function(X)
        is_inlier = np.ones(X.shape[0], dtype=int)
        is_inlier[scores < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Return anomaly score adjusted by offset_. 
        Score > 0 => inlier, < 0 => outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The shifted anomaly scores for each sample.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Compute the base iForest anomaly score for each sample, adjusting 
        path lengths if neighbor weighting is enabled.

        Higher scores => more normal (inlier), lower => more anomalous.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores for each sample.
        """
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        # If neighbor weighting, compute p(pos) from fitted k-NN
        if self.nn_ is not None:
            prob_pos = self.nn_.predict_proba(X)[:, 1]
            self.nn_weight_ = prob_pos
        else:
            self.nn_weight_ = np.zeros(X.shape[0], dtype=float)

        # Sum path lengths across all trees
        depths = np.zeros(X.shape[0], dtype=float)
        for tree, features in zip(self.estimators_, self.estimators_features_):
            # Select subset of features if needed
            if features is not None and len(features) < X.shape[1]:
                X_subset = X[:, features]
            else:
                X_subset = X

            leaves_index = tree.apply(X_subset)
            decision_path = tree.decision_path(X_subset).todense()

            n_nodes = decision_path.shape[1]
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            # Weighted path length
            list_arrays = []
            for i in range(n_nodes):
                path_col = np.asarray(decision_path[:, i]).ravel()
                obs_in_node = np.nonzero(path_col)[0]
                if self.type_weight == 'nearest_neighbors':
                    # Weighted by (1 - p(pos)) for each occupant
                    node_weights = 1.0 - self.nn_weight_[obs_in_node]
                    new_col = path_col.astype(float).copy()
                    new_col[obs_in_node] = node_weights
                    list_arrays.append(new_col)
                else:
                    list_arrays.append(path_col.astype(float))

            # Sum over columns
            if self.type_weight == 'nearest_neighbors':
                stacked = np.column_stack(list_arrays)
                sum_paths = np.sum(stacked, axis=1)
            else:
                sum_paths = np.sum(np.asarray(decision_path), axis=1)

            # Average path length correction
            depths += (
                sum_paths
                + _average_path_length(n_samples_leaf) * (1 - self.nn_weight_)
                - 1.0
            )

        self.depths_ = np.maximum(depths, 0)

        # Standard iForest formula
        c_val = _average_path_length([self.max_samples_])
        scores = 2.0 ** (-self.depths_ / (len(self.estimators_) * c_val))
        return scores


def _average_path_length(n_samples_leaf):
    """
    Compute the average path length for nodes with n_samples_leaf samples, 
    following the standard Isolation Forest logic.

    Parameters
    ----------
    n_samples_leaf : int or ndarray
        The number of samples in each leaf (node).

    Returns
    -------
    out : float or ndarray
        The average path length correction for each input leaf size.
    """
    arr = np.atleast_1d(n_samples_leaf)
    out = np.zeros_like(arr, dtype=float)

    # Cases:
    # - If node size <= 1, path length = 0
    # - If node size == 2, path length = 1
    # - Else the standard 2(H(n-1) - (n-1)/n)
    mask_1 = (arr <= 1)
    mask_2 = (arr == 2)
    not_mask = ~(mask_1 | mask_2)

    out[mask_1] = 0.0
    out[mask_2] = 1.0
    out[not_mask] = (
        2.0 * (np.log(arr[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (arr[not_mask] - 1.0) / arr[not_mask]
    )
    return out


##############################################################################
#                  PNN Class (PU + Noise Negative)
##############################################################################
class PNN(BaseEstimator, ClassifierMixin):
    """
    Two-step anomaly-based method for Positive and Unlabeled (PU) learning.
    It aims to detect hidden positives among unlabeled (assumed negatives) 
    and optionally removes or relabels them. Then it trains a base classifier
    (e.g. LogisticRegression, XGB) on the resulting data.

    Parameters
    ----------
    method : str or None, default=None
        Strategy to apply to detected anomalies:
          - 'removal': remove the anomalies from the training set
          - 'relabeling': mark them as positives
          - None: do not apply any special treatment
    treatment_ratio : float, default=0.10
        Fraction of negative samples to label as anomalies.
    anomaly_detector : object or None
        An object that implements .fit(X, y) and .score_samples(X).
        If None, a default WeightedIsoForest is used.
    high_score_anomaly : bool, default=False
        If True, high anomaly scores => anomalies.
        If False, low anomaly scores => anomalies.
    base_classifier : object or None, default=None
        Final classifier (must implement .fit and .predict). 
        If None, defaults to XGBClassifier.
    resampler : str or object, default='adasyn'
        Resampling method to handle imbalance. If 'adasyn', uses ADASYN. 
        Set to None to skip resampling.
    max_samples : 'auto' or int or float, default='auto'
        Passed to WeightedIsoForest if used.
    n_neighbors : int, default=5
        Passed to WeightedIsoForest if used.
    keep_treated : bool, default=True
        Whether to keep the treated data (Xt_, yt_) after fit.
    keep_final : bool, default=True
        Whether to keep the final training data (Xf_, yf_) after resampling.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    base_classifier_ : object
        The fitted base classifier.
    anomaly_detector_ : object
        The fitted anomaly detector.
    Xt_, yt_ : ndarray or None
        Treated data after anomaly removal/relabeling (None if dropped).
    Xf_, yf_ : ndarray or None
        Final data after optional resampling (None if dropped).
    removed_instances_, modified_instances_ : ndarray or None
        Indices of removed/relabelled anomalies.
    classes_ : ndarray
        Class labels used during training (from the base classifier).
    """

    def __init__(
        self,
        method=None,
        treatment_ratio=0.10,
        anomaly_detector=None,
        high_score_anomaly=False,
        base_classifier=None,
        resampler="adasyn",
        max_samples='auto',
        n_neighbors=5,
        keep_treated=True,
        keep_final=True,
        random_state=None
    ):
        self.method = method
        self.treatment_ratio = treatment_ratio
        self.anomaly_detector = anomaly_detector
        self.high_score_anomaly = high_score_anomaly
        self.base_classifier = base_classifier
        self.resampler = resampler
        self.max_samples = max_samples
        self.n_neighbors = n_neighbors
        self.keep_treated = keep_treated
        self.keep_final = keep_final
        self.random_state = random_state

    def fit(self, X, y):
        """
        1) Fit anomaly detector => find top anomalies
        2) Remove or relabel them
        3) Optionally resample
        4) Fit final classifier

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Binary labels {0,1}. 1 => known positive, 0 => unlabeled negative.

        Returns
        -------
        self : PNN
            The fitted instance.
        """
        if issparse(X) or issparse(y):
            logging.info("Converting sparse inputs to dense.")
            X = X.toarray() if issparse(X) else X
            y = y.toarray() if issparse(y) else y

        X = np.asarray(X)
        y = np.asarray(y)
        unique_y = np.unique(y)
        if not (len(unique_y) == 2 and 0 in unique_y and 1 in unique_y):
            raise ValueError(f"Labels must be in {{0,1}}, got {unique_y}.")

        # Copy data
        self.Xt_, self.yt_ = X.copy(), y.copy()
        self.Xf_, self.yf_ = None, None
        self.modified_instances_ = None
        self.removed_instances_ = None
        self.classes_ = None

        rnd_gen = check_random_state(self.random_state)

        # Default anomaly detector
        if self.anomaly_detector is None:
            self.anomaly_detector_ = WeightedIsoForest(
                n_estimators=100,
                n_neighbors=self.n_neighbors,
                max_samples=self.max_samples,
                random_state=rnd_gen.randint(np.iinfo(np.int32).max),
                n_jobs=-1
            )
        else:
            self.anomaly_detector_ = self.anomaly_detector

        # Default base classifier
        if self.base_classifier is None:
            self.base_classifier_ = XGBClassifier(
                random_state=rnd_gen.randint(np.iinfo(np.int32).max),
                n_jobs=-1
            )
        else:
            self.base_classifier_ = self.base_classifier

        # Resampler
        if self.resampler == 'adasyn':
            self.resampler_ = ADASYN(
                sampling_strategy=1.0,
                random_state=rnd_gen.randint(np.iinfo(np.int32).max)
            )
        else:
            self.resampler_ = self.resampler

        # ---- STEP 1: Anomaly detection ----
        if self.method in ['removal', 'relabeling']:
            try:
                if hasattr(self.anomaly_detector_, 'fit'):
                    # WeightedIsoForest expects .fit(X, pos_label=...)
                    self.anomaly_detector_.fit(self.Xt_, self.yt_)
                else:
                    logging.error("Anomaly detector has no .fit(...) method.")
                    return self
            except Exception as e:
                logging.error(
                    f"Anomaly detector {type(self.anomaly_detector_).__name__} error: {repr(e)}"
                )
                return self

            # Score only negatives
            ix_neg = np.where(self.yt_ == 0)[0]
            if len(ix_neg) == 0:
                logging.warning("No negative samples found, skipping anomaly detection.")
            else:
                X_neg = self.Xt_[ix_neg]
                score_func = getattr(self.anomaly_detector_, "score_samples", None)
                if score_func is None or not callable(score_func):
                    logging.error(
                        "Anomaly detector must implement .score_samples(...)."
                    )
                    return self

                scores = score_func(X_neg)
                if scores is None or len(scores) != len(ix_neg):
                    logging.error("Mismatch in anomaly scores shape.")
                    return self

                frac = self.treatment_ratio
                if self.high_score_anomaly:
                    # fraction => threshold is (1 - frac) quantile
                    thresh = np.quantile(scores, 1 - frac)
                    anom_idx = np.where(scores > thresh)[0]
                else:
                    # fraction => threshold is frac quantile
                    thresh = np.quantile(scores, frac)
                    anom_idx = np.where(scores < thresh)[0]

                self.modified_instances_ = ix_neg[anom_idx]

        # ---- STEP 2: Treatment (remove or relabel) ----
        if self.method == 'removal' and self.modified_instances_ is not None:
            self.removed_instances_ = self.modified_instances_
            if len(self.removed_instances_) > 0:
                self.Xt_ = np.delete(self.Xt_, self.removed_instances_, axis=0)
                self.yt_ = np.delete(self.yt_, self.removed_instances_, axis=0)

        elif self.method == 'relabeling' and self.modified_instances_ is not None:
            if len(self.modified_instances_) > 0:
                self.yt_[self.modified_instances_] = 1

        # ---- STEP 3: Resampling (optional) ----
        self.Xf_, self.yf_ = self.Xt_.copy(), self.yt_.copy()
        if self.resampler_ is not None and type_of_target(self.yf_) == 'binary':
            try:
                self.Xf_, self.yf_ = self.resampler_.fit_resample(self.Xt_, self.yt_)
            except Exception as e:
                logging.warning(
                    f"{type(self.resampler_).__name__} encountered an error: {repr(e)}"
                )
                self.Xf_, self.yf_ = self.Xt_.copy(), self.yt_.copy()

        # ---- STEP 4: Fit final classifier ----
        try:
            self.base_classifier_.fit(self.Xf_, self.yf_)
            if hasattr(self.base_classifier_, "classes_"):
                self.classes_ = self.base_classifier_.classes_.astype(int)
            else:
                self.classes_ = np.array([0, 1], dtype=int)
        except Exception as e:
            logging.error(f"Error fitting base classifier: {repr(e)}")

        # Optionally drop intermediate data
        if not self.keep_treated:
            self.Xt_, self.yt_ = None, None
        if not self.keep_final:
            self.Xf_, self.yf_ = None, None

        return self

    def predict(self, X):
        """
        Predict class labels {0,1} using the trained base classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels {0, 1}.
        """
        if not hasattr(self, "base_classifier_"):
            raise NotFittedError("PNN is not fitted. Call 'fit' first.")
        return self.base_classifier_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities using the trained base classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Predicted probabilities for each class.
        """
        if not hasattr(self, "base_classifier_"):
            raise NotFittedError("PNN is not fitted. Call 'fit' first.")
        if hasattr(self.base_classifier_, "predict_proba"):
            return self.base_classifier_.predict_proba(X)
        raise NotFittedError("Base classifier has no predict_proba method.")

    def get_params(self, deep=True):
        """Return a dictionary of parameters for GridSearchCV etc."""
        return {
            'method': self.method,
            'treatment_ratio': self.treatment_ratio,
            'anomaly_detector': self.anomaly_detector,
            'high_score_anomaly': self.high_score_anomaly,
            'base_classifier': self.base_classifier,
            'resampler': self.resampler,
            'max_samples': self.max_samples,
            'n_neighbors': self.n_neighbors,
            'keep_treated': self.keep_treated,
            'keep_final': self.keep_final,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set parameters (for GridSearchCV etc.)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self


##############################################################################
#                  ReliableValues
##############################################################################
class ReliableValues:
    """
    Detect anomalies among negative samples and yield boolean masks for 
    'reliable' negatives and positives.

    Parameters
    ----------
    method : str, default='removal'
        How to treat detected anomalies. Options:
          - 'removal': remove them from the negative set
          - 'relabeling': change their label to positive
    treatment_ratio : float, default=0.10
        Fraction of negative samples to label as anomalies.
    anomaly_detector : object or None, default=None
        An object with .fit() and .score_samples() methods (e.g., WeightedIsoForest).
    high_score_anomaly : bool, default=True
        If True, higher anomaly scores => anomalies; else lower scores => anomalies.
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    Xt_, yt_ : ndarray
        Copies of the original data passed to get_reliable.
    modified_instances_ : ndarray or None
        Indices of detected anomalies in the negative set.
    """

    def __init__(
        self,
        method='removal',
        treatment_ratio=0.10,
        anomaly_detector=None,
        high_score_anomaly=True,
        random_state=42
    ):
        self.method = method
        self.treatment_ratio = treatment_ratio
        self.anomaly_detector = anomaly_detector
        self.high_score_anomaly = high_score_anomaly
        self.random_state = random_state

        # Will be set during get_reliable(...)
        self.Xt_ = None
        self.yt_ = None
        self.modified_instances_ = None

    def get_reliable(self, X, y):
        """
        Identify anomalies among negative samples and produce boolean masks 
        for reliable negatives and positives.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            Binary labels {0,1}, where 1 => known positive, 0 => unlabeled negative.

        Returns
        -------
        reliable_neg_mask, reliable_pos_mask : (ndarray, ndarray)
            Boolean masks indicating reliable negatives and reliable positives.
        """
        if issparse(X) or issparse(y):
            logging.info("Converting sparse inputs to dense.")
            X = X.toarray() if issparse(X) else X
            y = y.toarray() if issparse(y) else y

        X = np.asarray(X)
        y = np.asarray(y)
        unique_y = np.unique(y)
        if not (len(unique_y) == 2 and 0 in unique_y and 1 in unique_y):
            raise ValueError(f"Labels must be in {{0,1}}, got {unique_y}.")

        # Keep copies
        self.Xt_ = X.copy()
        self.yt_ = y.copy()

        rnd_gen = check_random_state(self.random_state)

        # Fit the anomaly detector
        try:
            if hasattr(self.anomaly_detector, 'fit'):
                self.anomaly_detector.fit(self.Xt_, self.yt_)
            else:
                logging.error("Anomaly detector has no .fit(...) method.")
                return None, None
        except Exception as e:
            logging.error(f"Anomaly detector error: {repr(e)}")
            return None, None

        # Identify negative samples
        ix_neg = np.where(self.yt_ == 0)[0]
        if len(ix_neg) == 0:
            logging.warning("No negative samples found; skipping anomaly detection.")
            # All positives are 'reliable_pos', no reliable negatives
            reliable_neg_mask = np.zeros(len(self.yt_), dtype=bool)
            reliable_pos_mask = (self.yt_ == 1)
            return reliable_neg_mask, reliable_pos_mask

        # Score negative samples
        X_neg = self.Xt_[ix_neg]
        score_func = getattr(self.anomaly_detector, "score_samples", None)
        if not callable(score_func):
            logging.error("Anomaly detector must implement .score_samples(...).")
            return None, None

        scores = score_func(X_neg)
        if scores is None or len(scores) != len(ix_neg):
            logging.error("Mismatch in anomaly scores shape.")
            return None, None

        # Determine anomalies
        frac = self.treatment_ratio
        if self.high_score_anomaly:
            thresh = np.quantile(scores, 1 - frac)
            anom_idx = np.where(scores > thresh)[0]
        else:
            thresh = np.quantile(scores, frac)
            anom_idx = np.where(scores < thresh)[0]

        self.modified_instances_ = ix_neg[anom_idx]

        # Initialize masks with original labels
        reliable_neg_mask = (self.yt_ == 0)
        reliable_pos_mask = (self.yt_ == 1)

        # Adjust based on method
        if self.method == 'removal':
            # Mark anomalies as unreliable negatives
            reliable_neg_mask[self.modified_instances_] = False
        elif self.method == 'relabeling':
            # Convert anomalies to positives
            reliable_neg_mask[self.modified_instances_] = False
            reliable_pos_mask[self.modified_instances_] = True
        else:
            logging.warning(
                f"Unknown method '{self.method}'. Returning original label masks."
            )

        return reliable_neg_mask, reliable_pos_mask
