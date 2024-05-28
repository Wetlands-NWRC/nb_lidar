import time

import ee
import ee.batch


def monitor_task(task: ee.batch.Task) -> int:
    while task.status()["state"] in ["READY", "RUNNING"]:
        time.sleep(5)

    status_code = {"COMPLETED": 0, "FAILED": 1, "CANCELLED": 2}

    return status_code[task.status()["state"]]


def get_assets(parent: str, suffix: str = None) -> list[str]:
    assets = ee.data.listAssets({"parent": parent})["assets"]
    if suffix is not None:
        return [_.get("id") for _ in assets if _.get("id").endswith(suffix)]
    return [_.get("id") for _ in assets]


def classification_to_cloud_stroage(
    image, bucket, file_name, region, start: bool = False
) -> None:
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description="",
        bucket=bucket,
        fileNamePrefix=file_name,
        region=region,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileDimensions=[2048, 2048],
        skipEmptyTiles=True,
        scale=10,
    )

    if start:
        task.start()
    return task


def confusion2collection(confusion_matrix) -> ee.FeatureCollection:
    features = [
        ee.Feature(None, {"confusion_matrix": confusion_matrix.array()}),
        ee.Feature(None, {"order": confusion_matrix.order()}),
        ee.Feature(
            None, {"producers": confusion_matrix.producersAccuracy().toList().flatten()}
        ),
        ee.Feature(
            None, {"consumers": confusion_matrix.consumersAccuracy().toList().flatten()}
        ),
        ee.Feature(None, {"overall": confusion_matrix.accuracy()}),
    ]

    return ee.FeatureCollection(features)


def confusion_matrix_to_drive(
    confusion_matrix, folder, name, start: bool = True
) -> ee.batch.Task:
    if not isinstance(confusion_matrix, ee.confusionmatrix.ConfusionMatrix):
        raise TypeError("Must be a confusion matrix")
    collection = confusion2collection(confusion_matrix)
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description="",
        folder=folder,
        fileNamePrefix=name,
        fileFormat="GeoJSON",
    )

    if start:
        task.start()
    return task
