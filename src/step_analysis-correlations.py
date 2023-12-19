"""Correlation analysis

* Currently categorical ones

"""
import os
from os import path
import argparse

# Data manipulation
import pandas as pd
from pandas import DataFrame
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn import metrics
import math 

import wandb
from PIL import Image
from .utils import get_dataset
from .utils_plots import define_color_palette, plot_correlation_matrix
from itertools import product
from scipy import stats
import math

import warnings
from sklearn.exceptions import UndefinedMetricWarning

import settings
BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT
CONFIG_PIPELINE=settings.CONFIG_PIPELINE
STEP_NAME="analysis-correlations"

logger = settings.logging.getLogger(__name__)
settings.logging.getLogger('matplotlib').setLevel(level='INFO') # too much DEBUG msgs from plt
settings.logging.getLogger('urllib3').setLevel(level='INFO') # too much DEBUG msgs from it
#region local_libs ========================================


#endregion

#region main_code =========================================================
class AnalyzerCorrelations( ):

    COLOR_PALETTE_MODELS = []
    ds = None
    categorial_vars1 = []
    categorial_vars2 = []
    detector_rename = {
        'detector_chatgpt_qa_detector': 'ChatGPT_QA',
        'detector_xlmr_chatgptdetect_noisy': 'XLMR_ChatGPT',
        'detector_llm_det': 'LLMDet',
        'detector_gpt_zero': 'GPT_Zero',
        'detector_radar_vicuna_7B': 'Radar_Vicuna7B'
    }
    # detector_rename = {f"{k.replace('detector_', '')}_label":v for k,v in detector_rename.items()}        
    detector_rename = {f"{k}_label":v for k,v in detector_rename.items()}

    rs_ds = []

    def __init__(self, ds_txts_fpath:str, ds_writtersattitude_fpath:str, **kwargs) -> None:
        self.ds = self._preprocess(self.get_ds(ds_txts_fpath, ds_writtersattitude_fpath))

    def get_ds(self, ds_fpath, ds_writtersattitude_fpath, **kwargs) -> DataFrame:
        print(os.getcwd())
        ds_txts = pd.read_csv(ds_fpath, sep="\t")
        if ds_txts.columns[0].startswith('Unamed'):
            ds_txts = pd.read_csv(ds_fpath, sep="\t", index_col=[0])

        # load Writers attitudes
        ds_attitude = pd.read_csv(ds_writtersattitude_fpath, sep="\t", index_col=[0])
        if "_non_persuasive" in list(ds_attitude.columns):
            ds_attitude.rename({"_non_persuasive": "persuasiveness_non_persuasive"}, axis=1, inplace=True)
        if "_persuasive" in list(ds_attitude.columns):
            ds_attitude.rename({"_persuasive": "persuasiveness_persuasive"}, axis=1, inplace=True)


        # assign top1 label for each 
        for atti in ["emotion_", "irony_", "convincingness_", "persuasiveness_"]:
            ds_attitude[f"{atti}top1_label"] = ds_attitude[[c for c in ds_attitude.columns if (atti in c and not c.endswith("top1_label"))]].idxmax(axis=1)
            ds_attitude[f"{atti}top1_score"] = ds_attitude[[c for c in ds_attitude.columns if (atti in c and not c.endswith("top1_label"))]].max(axis=1)
            ds_attitude[f"{atti}top1_label"] = ds_attitude[f"{atti}top1_label"].apply(lambda v: v[v.find("_")+1:])

        ds = pd.concat([ds_txts, ds_attitude], axis=1)
        ds = ds[ds.tokeep==1]

        ds["model"] = ds["model"].map(lambda v: v.split('/')[-1])        


        return ds
    
    def _preprocess(self, df):
        # self.categorial_vars1 = ["model"] + [c for c in df.columns if c.startswith("detector_") and c.endswith("_label")]
        self.categorial_vars1 = [c for c in df.columns if c.startswith("detector_") and c.endswith("_label")]
        self.categorial_vars2 = [c for c in df.columns if not c.startswith("detector_") and c.endswith("_label")]    

        return df
    
    def _rename_detector(self, txt):
        return self.detector_rename.get(txt, txt)

    def compute_chi_test(self, df, vars1=[], vars2=[]):
        vars1 = vars1 or self.categorial_vars1
        vars2 = vars2 or self.categorial_vars2
        rs = []
        for i in list(product(vars1,vars2, repeat = 1)):
            if i[0] != i[1]:
                rs.append((i[0],i[1],
                            list(stats.chi2_contingency(pd.crosstab(df[i[0]], df[i[1]])))[1]))
                
        rs = pd.DataFrame(rs, columns = ['var1', 'var2','coeff'])
        rs = rs.pivot(index='var1', columns='var2', values='coeff') # -> into a crosstab
        return rs    
    

    def plot_correlations_cat_general(self, df=None, alpha=0.05):
        if df is None:
            df = self.ds
        rs = self.compute_chi_test(df[["category"] + self.categorial_vars1 + self.categorial_vars2], self.categorial_vars1, self.categorial_vars2)
        print(rs.shape)
        print(rs.columns)
        fig = plot_correlation_matrix(rs, significance_level=alpha)
        ax = fig.get_axes()[0]
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([lab.get_text().replace("_top1_label", "") for lab in ax.get_xticklabels()])
        # ax.set_yticklabels([lab.get_text().replace("_label", "") for lab in ax.get_yticklabels()])
        ax.set_yticklabels([self._rename_detector(lab.get_text()) for lab in ax.get_yticklabels()])
        fig.subplots_adjust(left=0.25)
        fig.suptitle(f"Correlations on categorical features. Showing p-values of Chi-test, (colored: p-value > {alpha}).", fontsize=14)
        fig.tight_layout()
        rs["group"] = "ALL"
        self.rs_ds.append(rs)
        return {f"correlations_categorical_general": fig}

    def plot_correlations_cat_bytask(self, df=None, alpha=0.05, groups=["category"], title="Correlations on categorical features by task"):
        if df is None:
            df = self.ds        

        dftmp = df[~df[groups[0]].isna()] #df[~df.category.isna()]
        n_figs = len(dftmp[groups[0]].unique()) #len(dftmp.category.unique())
        n_cols=3
        gs = plt.GridSpec(nrows=math.ceil(n_figs/n_cols), ncols=n_cols, width_ratios=[1]*n_cols, wspace=0.1, hspace=0.12)
        fig = plt.figure(figsize=(25, gs.nrows*6)) # figsize=(25, 15)

        for i, (grpk, grp_df) in enumerate(df.groupby(groups)):
            rs = self.compute_chi_test(grp_df[self.categorial_vars1 + self.categorial_vars2], self.categorial_vars1, self.categorial_vars2)
            print(rs.shape)
            print(rs.columns)
            print(rs)            
            # display(rs)
            ax = fig.add_subplot(gs[int(i/n_cols), (i%n_cols)])
            plot_correlation_matrix(rs, significance_level=alpha, fig=fig)#, cbar=(i%n_cols)) #only leave last column cbar
            ax.set_title(grpk, fontsize=15)    
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([lab.get_text().replace("_top1_label", "") for lab in ax.get_xticklabels()])
            # ax.set_yticklabels([lab.get_text().replace("_label", "") for lab in ax.get_yticklabels()])
            ax.set_yticklabels([self._rename_detector(lab.get_text()) for lab in ax.get_yticklabels()])
            if gs.nrows>1 and (i-1)<=gs.nrows: # only leave last row xticks            
                ax.set_xticklabels([])
            if (i%n_cols)!=0: # only leave first column yticks
                ax.set_yticklabels([])

            rs["group"] = grpk
            self.rs_ds.append(rs)                

        # fig.subplots_adjust(top=0.93)
        fig.subplots_adjust(top=0.85, bottom=0.2)
        fig.suptitle(f"{title}. Showing p-values of Chi-test, (colored: p-value > {alpha}).", fontsize=16)
        fig.tight_layout()

        return {f"correlations_categorical_bytask": fig}

#endregion

import io

def main(log_wandb, localtest, ds_version, dswatti_version, ds_artifact_name, t10sec, **kwargs):
    tgt_dir_files = path.join(BASE_DIR, "out")
    tgt_dir_figs = path.join(BASE_DIR, "out", "figs")
    logger.info(locals())

    tags=None
    groups = ["category"]
    cols2keep=[]
    suffix=''
    if "ablation" in ds_artifact_name:
        kwargs["ablation_variables"] = "group"
        tags=["ablation"]
        groups = ["group"]
        cols2keep=["group"]
        suffix= '_'+'_'.join(tags)

    run = None #wandb.init() # provide is art
    if log_wandb:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type=STEP_NAME) # provide is art

    # load dataset
    if localtest:
        ds_fpath = f"{BASE_DIR}/artifacts/{ds_artifact_name}:{ds_version}/dataset.tsv"
        ds_writtersattitude_fpath = f"{BASE_DIR}/artifacts/writters_attitude:{dswatti_version}/merged_fmt.tsv"
    else:
        # use artifacts
        artifact, local_path = get_dataset(run, artifact_name=ds_artifact_name, ds_version=ds_version, **kwargs)
        ds_fpath = path.join(local_path, artifact.files()[0].name)
        artifact, local_path = get_dataset(run, artifact_name="writters_attitude", ds_version=dswatti_version, **kwargs)
        ds_writtersattitude_fpath = path.join(local_path, "merged_fmt.tsv")


    logger.debug(ds_fpath)
    logger.debug(ds_writtersattitude_fpath)

    # compute evaluation metrics
    proc = AnalyzerCorrelations(ds_fpath, ds_writtersattitude_fpath, **kwargs)
    

    # generate plots 
    figs = {}
    figs.update( proc.plot_correlations_cat_general() )
    figs.update( proc.plot_correlations_cat_bytask(groups=groups, title="Correlations on categorical features by ablation scenario" if tags else "Correlations on categorical features by task") )

    df_rs =pd.concat(proc.rs_ds, axis=0)
    df_rs.index = df_rs.index.map(proc.detector_rename) # rename detectors
    rs_fpath = path.join(tgt_dir_files, f"correlation_analysis{suffix}.tsv")
    df_rs.to_csv(rs_fpath, sep='\t')

    if log_wandb:
        artifact_rs = wandb.Artifact(name=STEP_NAME, type="dataset")
        artifact_rs.add_file(local_path=rs_fpath, name="correlation_analysis.tsv")
        run.log_artifact(artifact_rs)

    # saving and logging plots
    for figname, fig in figs.items():
        logger.debug(f"Saving --> {figname}{suffix}")        
        img_fpath = path.join(tgt_dir_figs, f"{STEP_NAME}_{figname}{suffix}.png")
        fig.savefig(img_fpath)  # using step_name as grouper 
        if log_wandb:   
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            wandb.log({f"{STEP_NAME}_{figname}{suffix}".replace('_', '/'): wandb.Image(Image.open(buf))})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument("--ds_version", default="latest", type=str, help="version of dataset with detector predictions")
    parser.add_argument("--dswatti_version", default="latest", type=str, help="Writters attitude version")
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--localtest", action="store_true", help="ONLY use for local unit tests")    
    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)")
    parser.add_argument("--ds_artifact_name", default="ds_detection", type=str, help="name of the dataset to compute")

    args = parser.parse_args()

    main(**vars(args))
