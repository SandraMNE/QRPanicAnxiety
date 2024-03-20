"""Preprocess of Jopsephs detection files

"""
import os
import argparse
import wandb
import logging
import pandas as pd
from os import path
import re

env = os.getenv

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(env("LOGLEVEL", "INFO")))
print(logger)


import settings
BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT
CONFIG_PIPELINE=settings.CONFIG_PIPELINE
STEP_NAME="preprocess-dataset"



def preprocess_identifyvalid(df):
    """Sandra....

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    # removing missing texts and where the text is contains just punctuation (no digits nor letters)
    # df['context'] = df['context'].fillna('')
    pattern = r'[A-Za-z0-9]+'
    df['todrop'] = df['text'].apply(lambda x: 0 if re.search(pattern, str(x), re.IGNORECASE) else 1) #np.where(df_test.response.str.search(regex), 0, 1)
    df_problematic = df[(df.text.isna()) | (df.todrop==1)][['text','source','class']]
    def remove_problematic(row, df_problematic):
        df_sel = df_problematic.loc[(df_problematic.text==row['text']) & (df_problematic.source==row['source']) & (df_problematic['class']==row['class'])]
        if len(df_sel)>0:
            return 0
        else:
            return 1
    df['tokeep'] = df.apply(lambda row: remove_problematic(row,df_problematic),1)
    df.drop(['todrop'],1,inplace=True)

    return df


def main(file_or_dir, artifact_name:str, wandb_entity:str, wandb_project:str, log_wandb:bool, artifact_type:str="dataset", **kwargs):

    logger.debug(locals())
    assert file_or_dir
    tgt_dir_files = path.join(BASE_DIR, "out")


    fpaths = []
    if len(file_or_dir) ==1 and os.path.isdir(file_or_dir[0]):
        fpaths.append(file_or_dir[0])  # Add dataset directory to artifact
    elif len(file_or_dir)==1: # single file
        # raise "Not handled"
        fpaths.extend(file_or_dir)
        pass
    else:# multiple files
        fpaths.extend(file_or_dir)

    dss = []
    for f in fpaths:
        dftmp = pd.read_csv(f, sep=",")
        if dftmp.columns[0].startswith('Unamed'):
            dftmp = pd.read_csv(f, sep=",", index_col=[0])
        dftmp.reset_index(inplace=True, drop=True)
        dss.append(dftmp)


    df = pd.concat(dss, axis=0)
    print(df)

    # identify valid responses
    df = preprocess_identifyvalid(df)
    tgt_path = path.join(tgt_dir_files, f"{artifact_name}_preprocessed.tsv")
    df.to_csv(tgt_path, sep="\t")

    if log_wandb:
        run = wandb.init(project=wandb_project, entity=wandb_entity, job_type=STEP_NAME)
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(local_path=tgt_path, name="dataset.tsv")  # Add dataset directory to artifact
        run.log_artifact(artifact)  # Logs the artifact version "my_data:v0"

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    # parser.add_argument("file_or_dir", type=str, help="file or dir path to be added as artifact")
    parser.add_argument("file_or_dir", nargs='+', help="file or dir path to be added as artifact")
    parser.add_argument("--artifact_name", default="ds_paanx", type=str, help="Name that will be assigned to the artifact")
    parser.add_argument("--wandb_project", default=WANDB_PROJECT, type=str, help="project on wandb")
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY, type=str, help="team name on wandb")
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--artifact_type", type=str, default="dataset", help="dataset | model | ...")

    args = parser.parse_args()
    print(args)

    main(**vars(args))

