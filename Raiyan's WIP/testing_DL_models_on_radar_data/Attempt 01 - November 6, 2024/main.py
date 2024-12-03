import logging
from scripts.data_preprocessing import *
# from scripts.feature_extraction import *
# from scripts.train import *
# from scripts.evaluate import *

logging.basicConfig(filename='logs/main.log', level=logging.INFO)
logging.info("Starting pipeline...")

# Run each step in sequence
load_data(...)
# normalize_data(...)
# voxelize_data(...)
# create_sequences(...)
# train_model(...)
# evaluate_model(...)
