import ast
import pandas as pd
import wandb
import settings
import yaml
import os

BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT

def get_step_config(step_name):
    # get step config
    # try:
    with open(settings.CONFIG_PIPELINE, "r") as f:
        config_pipeline = yaml.load(f, Loader=yaml.FullLoader)
    config_step = config_pipeline.get("steps").get(step_name)   
    return config_step 
    # except Exception as err:
        # pass
        # return {}

def df_map_softmax_to_columns(df:pd.DataFrame, label_map:dict=None, prefix:str=None) -> pd.DataFrame:
    """Map classification softmax objects to columns

    Args:
        df (pd.DataFrame): dataframe with softmaxs
        label_map (dict, optional): Dict {old_name: new_name} to rename columns. Defaults to None.
        prefix (str, optional): prefix to be added to each column (e.g 'mytype_'). Defaults to None.

    Returns:
        Dataframe: modified dataframe        
    """

    _labels = ( [ ast.literal_eval(v)['label'] for v  in df.iloc[0,:].values if type(v)==str ])
    df = df.iloc[:,1:].rename(lambda n: f"{_labels[int(n)]}", axis="columns")
    df = df.applymap(lambda e: ast.literal_eval(e)['score'])
    if label_map:
        df.rename(columns=label_map, inplace=True)
    if prefix:
        df.rename(columns={c: f"{prefix}{c}" for c in df.columns}, inplace=True)
        
    return df

def get_dataset(run, artifact_name:str, ds_version:str="latest", wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT, **kwargs):
    # get version of the dataset

    # artifact_name="ds_detection"
    if run:
        artifact = run.use_artifact(f'{wandb_entity}/{wandb_project}/{artifact_name}:{ds_version}', type='dataset')
    else: # without a run
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_entity}/{wandb_project}/{artifact_name}:{ds_version}")

    artifact_dir = artifact.download()
    return artifact, artifact_dir