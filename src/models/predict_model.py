import sys

import ee
import ee.image
from src.helpers.image_processing import process_and_stack_images

AOI_ID = "projects/nb-lidar/assets/aoi_nb_south"
BUCKET = "eerfpl-exports"


def classification_to_cloud_stroage(image, bucket, file_name, region) -> None:
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description="",
        bucket=bucket,
        fileNamePrefix=file_name,
        region=region,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileDimensions=[2048, 2048],
        skipEmptyTiles=True,
        scale=10
    )

    task.start()
    return


def predict(aoi_id: str, model_id: str, terrain_type) -> ee.Image:
    aoi = ee.FeatureCollection(aoi_id).geometry()
    stack = process_and_stack_images(aoi, terrain_type)
    
    model = ee.Classifier.load(model_id)
    
    return stack.classify(model)
    
# def main(args: list[str]):
#     if len(args) != 2:
#         raise RuntimeError
    
#     terrain_type, model_id = args
    
#     if terrain_type not in ['dtm', 'dsm', 'srtm']:
#         raise RuntimeError
    
    
#     aoi = ee.FeatureCollection(AOI_ID).geometry()
#     stack = process_and_stack_images(aoi, terrain_type)
    
#     model = ee.Classifier.load(model_id)
    
#     prediction = stack.classify(model)
#     name = f'{terrain_type}_prdiction'
#     file_name_prefix = f'nb-lidar/{name}/{name}-'
    
#     classification_to_cloud_stroage(
#         image=prediction,
#         bucket=BUCKET,
#         file_name=file_name_prefix,
#         region=aoi
#     )

#     return 0

# if __name__ == '__main__':
#     ee.Initialize(project='nb-lidar')
#     main(sys.argv[1:])
