import argparse
import json
from utils_train import * 



parser = argparse.ArgumentParser(description="Train BreastAgeNet model with a given configuration file.")
parser.add_argument('--config_name', type=str, default='config_v3', required=True, help='Name of the configuration file (without .json)')
args = parser.parse_args()

config_name = args.config_name
config_path = f"/scratch_tmp/users/k21066795/BreastAgeNet/configs/{config_name}.json"
with open(config_path, "r") as f:
    config = json.load(f)
    print(config)
    

train_CV(config)
