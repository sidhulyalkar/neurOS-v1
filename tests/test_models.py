import numpy as np

from neuros.models import (
    EEGNetModel,
    CNNModel,
    RandomForestModel,
    SVMModel,
    KNNModel,
    GBDTModel,
    DinoV3Model,
)


def _train_and_predict(model_cls):
    model = model_cls()
    X = np.random.randn(20, 16)
    y = np.random.randint(0, 2, size=20)
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,)


def test_models_train_predict():
    # ensure each model can train and predict
    for cls in [EEGNetModel, CNNModel, RandomForestModel, SVMModel, KNNModel, GBDTModel, DinoV3Model]:
        _train_and_predict(cls)