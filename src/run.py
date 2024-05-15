import ee

from src.data.make_dataset import make_ee_dataset
from src.features.build_features import build_elevation_features
from src.models.train_and_assess import train_smile_random_forest_and_assess
from src.models.predict_model import make_prediction

TERRAIN_TYPE = 'dsm'  # dtm, srtm
TRAINING_ASSET_ID = ""
AOI_ASSET_ID = ""
LABEL_COL = "class_name"

def main():
    
    #~~~~~~~~~~~~~~~~~~#
    # Step 1: Make and pre process dataset
    make_ee_dataset(asset_id=TRAINING_ASSET_ID, label_col=LABEL_COL)
    
    # Step 2: Extract Features
    build_elevation_features(aoi_id=AOI_ASSET_ID, )


if __name__ == '__main__':
    main()