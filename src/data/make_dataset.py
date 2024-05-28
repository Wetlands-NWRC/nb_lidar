import ee
import json

from pathlib import Path

import ee.batch


ASSET_ID = "projects/nb-lidar/assets/nb_south_training"
OUT_NAME = "projects/nb-lidar/assets/nb_south_features"
LABEL_COL = "class_name"


def make_ee_dataset(asset_id: str, label_col: str, out_name: str):
    features = ee.FeatureCollection(asset_id)

    # remapl labels to int
    old_labels = features.aggregate_array(label_col).distinct()
    new_labels = ee.List.sequence(1, old_labels.size())

    features = features.remap(old_labels, new_labels, label_col)

    # add a random column for tain and test assignment
    features = features.randomColumn()

    # write out a lookup

    # send to to reference folder
    data = ee.Dictionary.fromLists(old_labels, new_labels).getInfo()

    current_location = Path(__file__).absolute()
    destination_location = current_location.parent.parent.parent / Path(
        "references/lookup.json"
    )
    with open(destination_location, "w") as fh:
        json.dump(data, fh, indent=4)

    task = ee.batch.Export.table.toAsset(
        collection=features, assetId=out_name, description=""
    )

    task.start()

    return


if __name__ == "__main__":
    ee.Initialize(project="nb-lidar")
    make_ee_dataset(ASSET_ID, LABEL_COL, OUT_NAME)
