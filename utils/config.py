import os
import yaml

CONFIG_FILE = "assets/config.yml"
if os.environ.get("CONFIG_FILE"):
    CONFIG_FILE = os.environ.get("CONFIG_FILE")

cf = yaml.safe_load(open(CONFIG_FILE))

cf["INSTANCE_CONNECTION_NAME"] = os.environ.get("INSTANCE_CONNECTION_NAME")