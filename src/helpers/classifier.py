import ee
import ee.batch

from src.utils.datautils import monitor_task


class SmileRandomForest:

    def __init__(self, ntrees: int = 10) -> None:
        self.ntrees = ntrees
        self.model = None

    def train(
        self,
        features: ee.FeatureCollection,
        class_property: str,
        input_properties: list[str] | ee.List,
    ) -> None:
        if self.model is not None:
            raise RuntimeError("The model has already been trained.")
        self.class_property = class_property
        self.model = ee.Classifier.smileRandomForest(self.ntrees).train(
            features=features,
            classProperty=class_property,
            inputProperties=input_properties,
        )
        return

    def predict(self, X: ee.Image | ee.FeatureCollection):
        if self.model is None:
            raise ValueError("You must train the model before predicting")
        return X.classify(self.model)

    def error_matrix(
        self, test, name: str = None, order: list[int] = None
    ) -> ee.confusionmatrix.ConfusionMatrix:
        name = name or "classification"
        test = test.classify(self.model)
        return test.errorMatrix(self.class_property, name, order)

    def save(self, asset_id: str):
        task = ee.batch.Export.classifier.toAsset(
            classifier=self.model, assetId=asset_id
        )
        task.start()

        return task

    @staticmethod
    def load(model_id: str):
        loaded_model = ee.Classifier.load(model_id)
        instance = SmileRandomForest()
        instance.model = loaded_model
        return instance


def train_smile_random_forest(
    train_features, class_property, predictors: list[str], **hyper_prams
) -> ee.classifier.Classifier:
    """creates and trains rf model"""
    default_hypter = {}

    default_hypter.update(hyper_prams)

    rf_model = ee.Classifier.smileRandomForest(**default_hypter).train(
        train_features, class_property, predictors
    )

    return rf_model


def _matrix_to_table(confusion_matrix: ee.confusionmatrix.ConfusionMatrix):
    pass


def assess_model(
    model: ee.classifier.Classifier, test: ee.FeatureCollection
) -> ee.FeatureCollection:
    pass


def predict(model: str | ee.classifier.Classifier, image: ee.Image):
    if isinstance(model, str):
        model = ee.Classifier.load(model)
    return image.classify(model)


def train_and_assess_model(features):
    train = features.filter("random <= 0.7")
    test = features.filter("random > 0.7")
    model = train_smile_random_forest()
    error_matrix = assess_model
