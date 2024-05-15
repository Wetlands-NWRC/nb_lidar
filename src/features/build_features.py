import sys

import ee
import ee.batch


from src.utils.datautils import monitor_task, table_to_asset
from src.helpers.image_processing import process_and_stack_images


def build_elevation_features(aoi_id: str, features_id: str, terrain_type: str, dest_id: str, image: ee.Image = None) -> None:

    product_type = terrain_type.lower()

    aoi = ee.FeatureCollection(aoi_id)  # constant
    input_features = ee.FeatureCollection(features_id)  # constant

    stack = image or process_and_stack_images(aoi, product_type)

    samples = stack.sampleRegions(
        collection=input_features, scale=10, tileScale=16, geometries=True
    )

    feature_task = ee.batch.Export.table.toAsset(collection=samples, assetId=dest_id, description="")

    feature_task.start()

    print(f"Exporting Features: {dest_id}")
    status_code = monitor_task(feature_task)
    
    if status_code > 0:
        raise RuntimeError(f"An error occurred while exporting features. Exit code: {status_code}")

    return


# def main(args: list[str]):
#     if len(args) != 1:
#         raise RuntimeError("Invalid number of arguments. Expected 1 argument.")
    
#     terrain_type = args[0]
    
#     if terrain_type not in ['dtm', 'dsm', 'srtm']:
#         raise RuntimeError("Not A Valid terrain type")
    
#     build_elevation_features(AOI_ID, FEATURES_ID, terrain_type)

#     return

# if __name__ == '__main__':
#     ee.Initialize(project='nb-lidar')
#     main(sys.argv[1:])