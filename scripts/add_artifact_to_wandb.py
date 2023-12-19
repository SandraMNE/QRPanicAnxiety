import os
import argparse
import wandb
import logging

env = os.getenv

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(env("LOGLEVEL", "INFO")))
print(logger)

WANDB_ENTITY=env("WANDB_ENTITY", None)
WANDB_PROJECT=env("WANDB_PROJECT", None)

def main(file_or_dir, artifact_name:str, wandb_entity:str, wandb_project:str, artifact_type:str="dataset", **kwargs):

    logger.debug(locals())
    assert file_or_dir
    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="add-artifact")
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    if len(file_or_dir) ==1 and os.path.isdir(file_or_dir[0]):
        artifact.add_dir(local_path=file_or_dir)  # Add dataset directory to artifact
    elif len(file_or_dir)==1: # single file
        # artifact.add_file(local_path=file_or_dir, name=f"{artifact_type}.{os.path.splitext(file_or_dir)[1]}") # if a single file normalize name and keep extension (e.g., dataset.tsv)
        artifact.add_file(local_path=file_or_dir[0], name=(f"{artifact_type}.{os.path.splitext(file_or_dir[0])[1]}").replace("..",".")) # if a single file normalize name and keep extension (e.g., dataset.tsv)
    else:# multiple files
        for fpath in file_or_dir:
            artifact.add_file(local_path=fpath)
    run.log_artifact(artifact)  # Logs the artifact version "my_data:v0"

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    # parser.add_argument("file_or_dir", type=str, help="file or dir path to be added as artifact")
    parser.add_argument("file_or_dir", nargs='+', help="file or dir path to be added as artifact")
    parser.add_argument("--artifact_name", type=str, help="Name that will be assigned to the artifact")
    parser.add_argument("--wandb_project", default=WANDB_PROJECT, type=str, help="project on wandb")
    parser.add_argument("--wandb_entity", default=WANDB_ENTITY, type=str, help="team name on wandb")
    parser.add_argument("--artifact_type", type=str, default="dataset", help="dataset | model | ...")

    args = parser.parse_args()

    main(**vars(args))

