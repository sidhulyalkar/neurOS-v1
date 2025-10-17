"""
Ensemble methods and model stacking for improved BCI performance.

Combines multiple models to achieve better accuracy and robustness
than individual models alone.
"""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_predict

from neuros.models.base_model import BaseModel
from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.random_forest_model import RandomForestModel


class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.

    Supports both hard voting (majority vote) and soft voting
    (averaged probabilities).

    Parameters
    ----------
    models : list of BaseModel
        List of trained or untrained models
    voting : str
        Voting strategy: 'hard' or 'soft' (default: 'soft')
    weights : array-like
        Weights for each model (default: None, equal weights)
    """

    def __init__(self, models: List[BaseModel], voting: str = 'soft',
                 weights: Optional[np.ndarray] = None):
        super().__init__()
        self.models = models
        self.voting = voting
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using voting."""
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = np.array([model.predict(X) for model in self.models])
            # Weighted vote
            n_samples = X.shape[0]
            n_classes = int(np.max(predictions) + 1)

            voted = np.zeros((n_samples, n_classes))
            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j in range(n_samples):
                    voted[j, int(pred[j])] += weight

            return np.argmax(voted, axis=1)

        else:  # soft voting
            # Average predicted probabilities
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction (weighted average)."""
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        all_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                all_probs.append(model.predict_proba(X))
            else:
                # Convert hard predictions to probabilities
                pred = model.predict(X)
                n_classes = int(np.max(pred) + 1)
                probs = np.zeros((len(pred), n_classes))
                for i, p in enumerate(pred):
                    probs[i, int(p)] = 1.0
                all_probs.append(probs)

        # Weighted average
        weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
        return weighted_probs


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble with meta-learner.

    Uses predictions from base models as features for a meta-model
    that learns optimal combination weights.

    Parameters
    ----------
    base_models : list of BaseModel
        Base-level models
    meta_model : BaseModel
        Meta-level model (default: LogisticRegression)
    use_probas : bool
        Use predicted probabilities as meta-features (default: True)
    cv : int
        Number of cross-validation folds for meta-features (default: 5)
    """

    def __init__(self, base_models: List[BaseModel],
                 meta_model: Optional[BaseModel] = None,
                 use_probas: bool = True, cv: int = 5):
        super().__init__()
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else SimpleClassifier()
        self.use_probas = use_probas
        self.cv = cv

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train stacked ensemble."""
        # Train base models
        for model in self.base_models:
            model.train(X, y)

        # Generate meta-features using cross-validation
        # to avoid overfitting
        meta_features = self._generate_meta_features(X, y)

        # Train meta-model
        self.meta_model.train(meta_features, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Stacked prediction."""
        if not self.is_trained:
            raise RuntimeError("Stacking ensemble not trained")

        # Get predictions from base models
        meta_features = self._get_base_predictions(X)

        # Meta-model prediction
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Stacked probability prediction."""
        if not self.is_trained:
            raise RuntimeError("Stacking ensemble not trained")

        meta_features = self._get_base_predictions(X)
        return self.meta_model.predict_proba(meta_features)

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        n_samples = X.shape[0]

        if self.use_probas:
            # Use probability predictions
            meta_features_list = []
            for model in self.base_models:
                # For each model, get out-of-fold predictions
                # to avoid overfitting
                if hasattr(model, '_model') and hasattr(model._model, 'predict_proba'):
                    # Use sklearn's cross_val_predict if possible
                    probs = cross_val_predict(
                        model._model, X, y,
                        cv=self.cv,
                        method='predict_proba'
                    )
                    meta_features_list.append(probs)
                else:
                    # Fallback: use direct predictions
                    probs = model.predict_proba(X)
                    meta_features_list.append(probs)

            meta_features = np.hstack(meta_features_list)
        else:
            # Use class predictions
            meta_features = np.column_stack([
                model.predict(X) for model in self.base_models
            ])

        return meta_features

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from base models for meta-model input."""
        if self.use_probas:
            probs_list = [model.predict_proba(X) for model in self.base_models]
            return np.hstack(probs_list)
        else:
            preds = np.column_stack([model.predict(X) for model in self.base_models])
            return preds


class BoostingEnsemble(BaseModel):
    """
    Simplified boosting ensemble for BCI.

    Sequentially trains models, with each new model focusing on
    samples that previous models found difficult.

    Parameters
    ----------
    base_model_class : class
        Class of base model to use
    n_estimators : int
        Number of boosting iterations (default: 10)
    learning_rate : float
        Shrinkage parameter (default: 0.1)
    """

    def __init__(self, base_model_class, n_estimators: int = 10,
                 learning_rate: float = 0.1):
        super().__init__()
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []
        self.model_weights = []

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train boosted ensemble."""
        n_samples = X.shape[0]

        # Initialize weights uniformly
        sample_weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # Train new model on weighted samples
            model = self.base_model_class()

            # Sample according to weights
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            X_sampled = X[indices]
            y_sampled = y[indices]

            model.train(X_sampled, y_sampled)

            # Calculate error
            predictions = model.predict(X)
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect)

            if error > 0.5:
                # Model is worse than random, stop
                break

            # Calculate model weight
            if error == 0:
                model_weight = 1.0
            else:
                model_weight = 0.5 * np.log((1 - error) / error)

            # Update sample weights
            sample_weights *= np.exp(model_weight * incorrect.astype(float))
            sample_weights /= np.sum(sample_weights)

            self.models.append(model)
            self.model_weights.append(model_weight)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Boosted prediction."""
        if not self.is_trained or len(self.models) == 0:
            raise RuntimeError("Boosting ensemble not trained")

        # Weighted voting
        n_samples = X.shape[0]
        predictions = np.array([model.predict(X) for model in self.models])

        # Compute weighted vote
        n_classes = int(np.max(predictions) + 1)
        votes = np.zeros((n_samples, n_classes))

        for pred, weight in zip(predictions, self.model_weights):
            for i in range(n_samples):
                votes[i, int(pred[i])] += self.learning_rate * weight

        return np.argmax(votes, axis=1)


class BaggingEnsemble(BaseModel):
    """
    Bootstrap aggregating (bagging) ensemble.

    Trains multiple models on bootstrap samples and averages predictions.
    Reduces variance and improves stability.

    Parameters
    ----------
    base_model_class : class
        Class of base model to use
    n_estimators : int
        Number of models in ensemble (default: 10)
    max_samples : float
        Fraction of samples to use for each model (default: 1.0)
    """

    def __init__(self, base_model_class, n_estimators: int = 10,
                 max_samples: float = 1.0):
        super().__init__()
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self.models = []

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train bagged ensemble."""
        n_samples = X.shape[0]
        n_bootstrap = int(n_samples * self.max_samples)

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_bootstrap, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train model
            model = self.base_model_class()
            model.train(X_boot, y_boot)

            self.models.append(model)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Bagged prediction (majority vote)."""
        if not self.is_trained or len(self.models) == 0:
            raise RuntimeError("Bagging ensemble not trained")

        predictions = np.array([model.predict(X) for model in self.models])

        # Majority vote
        n_samples = X.shape[0]
        n_classes = int(np.max(predictions) + 1)

        votes = np.zeros((n_samples, n_classes))
        for pred in predictions:
            for i in range(n_samples):
                votes[i, int(pred[i])] += 1

        return np.argmax(votes, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Bagged probability prediction (average)."""
        if not self.is_trained or len(self.models) == 0:
            raise RuntimeError("Bagging ensemble not trained")

        all_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                all_probs.append(model.predict_proba(X))
            else:
                # Convert to probabilities
                pred = model.predict(X)
                n_classes = int(np.max(pred) + 1)
                probs = np.zeros((len(pred), n_classes))
                for i, p in enumerate(pred):
                    probs[i, int(p)] = 1.0
                all_probs.append(probs)

        return np.mean(all_probs, axis=0)


def create_default_ensemble() -> VotingEnsemble:
    """
    Create a default ensemble with diverse models.

    Returns
    -------
    ensemble : VotingEnsemble
        Ensemble with SimpleClassifier, SVMModel, and RandomForest
    """
    models = [
        SimpleClassifier(),
        SVMModel(C=1.0, gamma='scale'),
        RandomForestModel(n_estimators=50)
    ]

    return VotingEnsemble(models=models, voting='soft')


def create_fast_ensemble() -> VotingEnsemble:
    """
    Create a fast ensemble optimized for low latency.

    Returns
    -------
    ensemble : VotingEnsemble
        Fast ensemble with multiple SimpleClassifier instances
    """
    models = [
        SimpleClassifier(),
        SimpleClassifier(),
        SVMModel(C=0.1, gamma='scale')  # Faster with lower C
    ]

    return VotingEnsemble(models=models, voting='soft')
