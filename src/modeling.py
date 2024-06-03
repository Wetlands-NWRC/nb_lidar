from typing import Any
import ee


def train_smile_random_forest(
    training_features,
    class_property,
    predictors,
    hyper_parameters: dict[str, Any] = None,
):
    """trains a smile random forest model

    Args:
        training_features (ee.FeatureCollection): features to train on
        class_property (str): class labels column
        predictors (list[str]|ee.List): List of predictors to use
        hyper_parameters (dict[str, Any], optional): options to create rf model. Defaults to None.

    Returns:
        ee.Classifier: trained ee.Classifier.smileRandomForest model
    """
    if hyper_parameters is None:
        hyper_parameters = {"numberOfTrees": 1000, "seed": 0}
    else:
        if "numberOfTrees" not in hyper_parameters:
            hyper_parameters["numberOfTrees"] = 1000
        if "seed" not in hyper_parameters:
            hyper_parameters["seed"] = 0

    return ee.Classifier.smileRandomForest(**hyper_parameters).train(
        training_features, class_property, predictors
    )


def _error_matrix_to_feature_collection(matrix) -> ee.FeatureCollection:
    features = [
        ee.Feature(None, {"confusion_matrix": matrix.array()}),
        ee.Feature(None, {"order": matrix.order()}),
        ee.Feature(None, {"producers": matrix.producersAccuracy().toList().flatten()}),
        ee.Feature(None, {"consumers": matrix.consumersAccuracy().toList().flatten()}),
        ee.Feature(None, {"overall": matrix.accuracy()}),
    ]

    return ee.FeatureCollection(features)


def assess_model(testing_features, model) -> ee.FeatureCollection:
    """Assess a smile random forest model and converts to feature collection

    Args:
        testing_features (ee.FeatureCollection): features to test the model against
        model (ee.Classifier): the Random Forest model you want to assess

    Returns:
        ee.FeatureCollection: _description_
    """
    label_col = "class_name"
    order = testing_features.aggregate_array(label_col).distinct()
    predict = testing_features.classify(model)
    error_matrix = predict.errorMatrix(label_col, "classification", order)
    return _error_matrix_to_feature_collection(error_matrix)


def predict_model(image, model) -> ee.Image:
    """makes prediction on the image using the trained rf model"""
    return image.classify(model)
