import os
import logging

env = os.environ.get

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, 'src')) # '../'))
WANDB_DIR=os.path.join(BASE_DIR, ".wandb")
os.environ[WANDB_DIR]=WANDB_DIR

# ============

PROJECT_NAME='paanx'
WANDB_ENTITY=env("WANDB_ENTITY", "idsia-nlp")
WANDB_PROJECT=env("WANDB_PROJECT", "paanx")
CONFIG_PIPELINE=env("CONFIG_PIPELINE", os.path.join(BASE_DIR, "config_pipeline.yaml"))

# ============

## RELEASE_NUMBER = __import__('xxx').VERSION
## VERSION = RELEASE_NUMBER
## LOCAL_NETWORK = env('LOCAL_NETWORK', '127.0.0.0/8')

DEV_ENV = env('DEV_ENV', 'test')
LOGLEVEL = logging.getLevelName(env('LOGLEVEL', 'INFO'))
logging.basicConfig(level=LOGLEVEL)

