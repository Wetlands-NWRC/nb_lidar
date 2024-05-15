import sys
import ee
import ee.batch
import ee.batch


from src.helpers.classifier import SmileRandomForest
from src.utils.datautils import table_to_drive, monitor_task


def format_error_matrix_for_export(
    matrix: ee.confusionmatrix.ConfusionMatrix,
) -> ee.featurecollection.FeatureCollection:
    # TODO implemet helper to add producers consumers overall order
    features = [
        ee.Feature(None, {'confusion_matrix': matrix.array()}),
        ee.Feature(None, {'order': matrix.order()}),
        ee.Feature(None, {'producers': matrix.producersAccuracy().toList().flatten()}),
        ee.Feature(None, {'consumers': matrix.consumersAccuracy().toList().flatten()}),
        ee.Feature(None, {'overall': matrix.accuracy()}),
    ]

    return ee.FeatureCollection(features)


def train_smile_random_forest_and_assess(
    features: ee.FeatureCollection,
    n_trees: int = 1000,
    label_col: str = "class_name",
    remove_props: list = None,
    split_column: str = 'random',
    split: float = 0.7
) -> tuple[SmileRandomForest, ee.confusionmatrix.ConfusionMatrix]:

    to_remove = ["system:index", "random", "eco_region", "class_name"]
    if remove_props:
        to_remove.extend(remove_props)

    predictors = features.first().propertyNames().removeAll(to_remove)

    train = features.filter(f"{split_column} < {split}")
    test = features.filter(f"{split_column} >= {split}")

    model = SmileRandomForest(ntrees=n_trees)
    model.train(train, label_col, predictors)

    order = test.aggregate_array("class_name").distinct()
    confusion_matrix = model.error_matrix(test, order=order)

    return model, confusion_matrix


def save_model(model_name: str, model: SmileRandomForest) -> int:
    model_task = model.save(model_name)

    print(f"Exporting Model: {model_name}")
    status_code = monitor_task(model_task)
    if status_code > 0:
        raise RuntimeError(f"An error occurred while saving the model. Exit code: {status_code}")
    return status_code


def save_confusion_matrix_to_drive(obj, folder, name) -> None:
    if isinstance(obj, ee.confusionmatrix.ConfusionMatrix):
        obj = format_error_matrix_for_export(obj)

    task = ee.batch.Export.table.toDrive(
        collection=obj,
        description="",
        folder=folder,
        fileNamePrefix=name,
        fileFormat='GeoJSON'
    )
    
    task.start()
    return None

def main(args: list[str]) -> int:
    if len(args) != 1:
        raise RuntimeError("Invalid Number of Args. Must be exactly one argument")
    
    sample_id = args[0]
    basename = sample_id.split("/")[-1].split('_')[0]
    model_basename = f"{basename}_rf_model"
    model_name = "/".join(sample_id.split('/')[:-2]) + f'/models/{model_basename}'

    matrix_destination = 'nb_lidar_assessments'
    matrix_name = f'{model_basename}_confusion_matrix'

    features = ee.FeatureCollection(sample_id)

    model, cfm = train_smile_random_forest_and_assess(features=features)

    save_confusion_matrix_to_drive(cfm, folder=matrix_destination, name=matrix_name)
    save_model(model_name=model_name, model=model)
    
    return 0

if __name__ == '__main__':
    ee.Initialize(project='nb-lidar')
    main(sys.argv[1:])