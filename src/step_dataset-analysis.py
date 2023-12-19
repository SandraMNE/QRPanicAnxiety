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

import wandb
from PIL import Image
from .utils import get_dataset
from .utils_plots import hex_to_rgb, counts_for_plot, proportion_plot
from itertools import product

# sns.set_style("white") # darkgrid, whitegrid, dark, white, and ticks
# plt.figure(figsize=(7, 7))

import settings
BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT
CONFIG_PIPELINE=settings.CONFIG_PIPELINE
STEP_NAME="dataset-analysis"

logger = settings.logging.getLogger(__name__)
settings.logging.getLogger('matplotlib').setLevel(level='INFO') # too much DEBUG msgs from plt
settings.logging.getLogger('urllib3').setLevel(level='INFO') # too much DEBUG msgs from it
#region local_libs ========================================

def normalize(df:pd.DataFrame, cols = [], normtype="minmax"):
    # copy the data
    df_min_max_scaled = df.copy()
    
    # apply normalization techniques
    for column in (cols or df.columns): # df_min_max_scaled.columns:
        if normtype=='minmax':
            df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
        elif normtype=='max':
            df_min_max_scaled[column] = (df_min_max_scaled[column] ) / (df_min_max_scaled[column].abs().max())    
        elif normtype=="zscale":
                df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].mean()) / df_min_max_scaled[column].std()   
    
    # view normalized data
    return df_min_max_scaled

# def counts_for_plot(df:pd.DataFrame, var1, var2, relative_to_var1_value:str=None):
#     dftmp = df[[var1, var2]].melt(id_vars=[var1])
#     dftmp = dftmp.groupby(by=[var1, "value"]).count().reset_index()    
#     if relative_to_var1_value:
#         anchor_values = dict(dftmp[dftmp[var1]==relative_to_var1_value][["value", "variable"]].values)        
#         dftmp["variable"] = dftmp.apply(lambda e: e["variable"] - anchor_values.get(e["value"], 0), axis=1)
#         dftmp = ( dftmp.query(f"{var1}!='{relative_to_var1_value}'") ) # remove anchor entries
#     return dftmp

# def proportion_plot(labels, counts, **kwargs):    
#     palette_color = sns.color_palette('muted') 
#     plt.pie(counts, labels=labels, colors=palette_color, autopct='%.0f%%')  

# def hex_to_rgb(hex):
#   return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

#endregion

#region main_code =========================================================
class DatasetAnalysis( ):

    COLOR_PALETTE_MODELS = []
    ds = None

    def __init__(self, ds_txts_fpath:str, ds_writtersattitude_fpath:str, **kwargs) -> None:
        self.ds = self.get_ds(ds_txts_fpath, ds_writtersattitude_fpath, **kwargs)
        self.COLOR_PALETTE_MODELS = self.define_color_palette(self.ds, for_column="model", fixed={'human': '#7f7f7f'}) # human -> gray

    def get_ds(self, ds_txts_fpath, ds_writtersattitude_fpath, **kwargs) -> DataFrame:
        #TODO: use paths from params
        print(os.getcwd())
        # ds_txts = pd.read_csv("./data/data/detection_dataset/20231030_145712_databricks-dolly_size-1000_melted_stats.tsv", sep="\t", index_col=[0])
        # ds_attitude = pd.read_csv(path.join(".", "out", "merged_fmt.tsv"), sep="\t", index_col=[0])
        ds_txts = pd.read_csv(ds_txts_fpath, sep="\t")
        if ds_txts.columns[0].startswith('Unamed'):
            ds_txts = pd.read_csv(ds_txts_fpath, sep="\t", index_col=[0])
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

        if "ablation_variables" in kwargs:
             ds.rename({kwargs.get("ablation_variables"): "ablation_variables"}, axis=1, inplace=True) # rename ablation variables column 

        return ds

    def define_color_palette(self, ds:DataFrame=None, for_column="model", base_palette=sns.color_palette(), fixed={'human': '#7f7f7f'}): #1f77b4
        if ds is None:
            ds = self.ds        
        _models = ds[for_column].unique().tolist()
        _palette = base_palette.as_hex()
        color_palette_models = {}
        if fixed:
            for k,c in fixed.items():
                if c in _palette:
                    _palette.remove(c) #_palette.index(c)
                if k in _models:
                    _models.remove(k)        
        color_palette_models = {**{ m: c for m,c in zip(_models, _palette) }, **fixed}
        return color_palette_models

    def plot_proportions_general(self, ds:DataFrame=None) -> Figure:
        if ds is None:
            ds = self.ds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.tight_layout(pad=10.0)
        fig.suptitle('Dataset proportions')

        # Generators distribution
        dftmp = ds.groupby(['model']).size().reset_index(name='counts')
        ax1.set_title("Text source")
        ax1.pie(dftmp.counts.values, labels=dftmp.model.values, colors=[self.COLOR_PALETTE_MODELS[m] for m in dftmp.model.values], autopct='%.0f%%') 

        # Generators distribution
        dftmp = ds.groupby(['category']).size().reset_index(name='counts')
        ax2.set_title("Task")
        ax2.pie(dftmp.counts.values, labels=dftmp["category"].values, autopct='%.0f%%') 
        # fig.tight_layout()

        return {f"general_proportions": fig}
    
    def plot_proportions_attitude(self, ds:DataFrame=None) -> Figure:
        if ds is None:
            ds = self.ds
        g = sns.FacetGrid(col="variable", height=6, 
                        data= ds.melt(value_vars=[c for c in ds.columns if "_top1_label" in c]).groupby(['variable', 'value']).size().reset_index(name='counts'))
        g.map(proportion_plot, "value", "counts")
        g.fig.tight_layout()

        return {f"wattitude_proportions": g.figure}
    
    def counts_for_plot_ablation(self, ds, var1, var2, ablation_reference_group="uncontrolled", ablation_variable_column="ablation_variables", relative_to_var1_value=None) :
        dfs = []
        for i in list(product([ablation_reference_group], ds[ablation_variable_column].unique().tolist() , repeat = 1)):
            if i[0] != i[1]:
                dfi = counts_for_plot(ds.query(f"{ablation_variable_column}=='{i[0]}'"), var1, var2, relative_to_var1_value)
                dfi["ablation_variable"] = i[0]
                dfi["ablation_group"] = f"{i[0]}-{i[1]}"
                dfs.append(dfi)
                dfi = counts_for_plot(ds.query(f"{ablation_variable_column}=='{i[1]}'"), var1, var2, relative_to_var1_value)
                dfi["ablation_variable"] = i[1]
                dfi["ablation_group"] = f"{i[0]}-{i[1]}"
                dfs.append(dfi)
                pass

        return pd.concat(dfs, axis=0)
        
    
    def plot_counts_wattitude(self, ds:DataFrame=None, **kwargs):
        if ds is None:
            ds = self.ds

        ablation_vars = ds["ablation_variables"].unique().tolist() if "ablation_variables" in kwargs else None
        print(ablation_vars)
        figs={}        

        #region Emotion
        atti="emotions"
        # countplot        
        if ablation_vars:
            dati = self.counts_for_plot_ablation(ds, var1="model", var2="emotion_top1_label")
            for kgrp, grp in dati.groupby(by=['ablation_group']):
                # g = sns.catplot(data=dati, x="value", y="variable", hue="model", row="ablation_variable", col="ablation_group", kind="bar", height=10, aspect=2)
                g = sns.catplot(data=grp, x="value", y="variable", hue="model", col="ablation_variable", kind="bar", height=5, aspect=1.5, palette=self.COLOR_PALETTE_MODELS)
                fig = g.fig
                g.set_axis_labels(f"{atti}", "Counts")
                g.set_xticklabels(rotation=90) 
                fig.tight_layout()
                figs[f"wattitude_{atti}_counts_ablation_{kgrp}"]=fig
        else:
            dati = counts_for_plot(ds, var1="model", var2="emotion_top1_label")
            fig = plt.figure(figsize=(16, 8))
            sns.barplot(data=dati , y="variable", x="value", hue="model", palette=self.COLOR_PALETTE_MODELS )
            plt.xticks(rotation=90)
            plt.xlabel(f"{atti}")
            plt.ylabel("Counts")
            fig.tight_layout()
            figs[f"wattitude_{atti}_counts"]=fig

        # countplot without neutral
        if ablation_vars:
            dati = self.counts_for_plot_ablation(ds, var1="model", var2="emotion_top1_label").query("value!='neutral'")
            for kgrp, grp in dati.groupby(by=['ablation_group']):
                g = sns.catplot(data=grp, x="value", y="variable", hue="model", col="ablation_variable", kind="bar", height=5, aspect=1.5, palette=self.COLOR_PALETTE_MODELS)
                g.set_xticklabels(rotation=90) 
                fig = g.fig
                g.set_axis_labels(f"{atti} (without neutral)", "Counts")
                fig.tight_layout()
                figs[f"wattitude_{atti}_counts-noneutral_ablation_{kgrp}"]=fig
        else:        
            fig = plt.figure(figsize=(16, 8))
            sns.barplot(data= counts_for_plot(ds, var1="model", var2="emotion_top1_label").query("value!='neutral'"), y="variable", x="value", hue="model", palette=self.COLOR_PALETTE_MODELS )
            plt.xticks(rotation=90)
            plt.xlabel(f"{atti} (without neutral)")
            plt.ylabel("Counts")
            fig.tight_layout()
            figs[f"wattitude_{atti}_counts-noneutral"]=fig

        # countplot relative to human          
        if ablation_vars:
            dati = self.counts_for_plot_ablation(ds, var1="model", var2="emotion_top1_label", relative_to_var1_value="human")
            for kgrp, grp in dati.groupby(by=['ablation_group']):
                g = sns.catplot(data=grp, x="value", y="variable", hue="model", col="ablation_variable", kind="bar", height=5, aspect=1.5, palette=self.COLOR_PALETTE_MODELS)
                g.set_xticklabels(rotation=90) 
                fig = g.fig
                g.set_axis_labels(atti, "Counts relative to human")
                fig.tight_layout()
                figs[f"wattitude_{atti}_counts-relative2human_ablation_{kgrp}"]=fig
        else:        
            fig = plt.figure(figsize=(16, 8))
            sns.barplot(data=counts_for_plot(ds, var1="model", var2="emotion_top1_label", relative_to_var1_value="human"), y="variable", x="value", hue="model", palette=self.COLOR_PALETTE_MODELS )
            plt.xticks(rotation=90)
            plt.xlabel(f"{atti}")
            plt.ylabel("Counts relative to human")
            fig.tight_layout()
            figs[f"wattitude_{atti}_counts-relative2human"]=fig

        # signals barely present in Human texts
        tgt_emotions=["optimism", "excitement",  "remorse", "gratitude", "curiosity", "nervousness"]
        palette = sns.color_palette("husl")
        if ablation_vars:            
            dati = ds[ds.emotion_top1_label.isin(tgt_emotions)]
            # for kgrp, grp in dati.groupby(by=['ablation_group']):
            for abvar in ablation_vars:
                if abvar != "uncontrolled":
                    print(f"\n================= {abvar} ==============\n ")
                    for kgrp, grp in dati.query(f"ablation_variables=='uncontrolled' or ablation_variables=='{abvar}'").groupby(by=["model"]):
                        print(kgrp)
                        print(grp.emotion_top1_label.value_counts())
                        print("----------------------------\n")
                    g = sns.catplot(data=dati.query(f"ablation_variables=='uncontrolled' or ablation_variables=='{abvar}'"), 
                                    x="emotion_top1_score", y="model", hue="emotion_top1_label", kind="strip", height=5, aspect=1.5, palette=palette,
                                    col="ablation_variables", col_order=["uncontrolled", abvar])
                    g.set_xticklabels(rotation=90) 
                    fig = g.fig
                    # g.set_axis_labels(atti, "Counts relative to human")
                    fig.tight_layout()
                    figs[f"wattitude_{atti}_strip_ablation_uncontrolled-{abvar}"]=fig
        else:        
            fig = plt.figure(figsize=(15, 7))
            sns.stripplot(data=ds[ds.emotion_top1_label.isin(tgt_emotions)], y="model", x="emotion_top1_score", hue="emotion_top1_label", size=6
                        , palette=palette)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            fig.tight_layout()
            figs[f"wattitude_{atti}_strip"]=fig
        #endregion

        #region others: irony, convincingness
        for atti in ["irony", "convincingness", "persuasiveness"]:

            if ablation_vars:
                dati = self.counts_for_plot_ablation(ds, var1="model", var2=f"{atti}_top1_label")
                for kgrp, grp in dati.groupby(by=['ablation_group']):
                    # g = sns.catplot(data=dati, x="value", y="variable", hue="model", row="ablation_variable", col="ablation_group", kind="bar", height=10, aspect=2)
                    g = sns.catplot(data=grp, x="value", y="variable", hue="model", col="ablation_variable", kind="bar", height=5, aspect=1.5, palette=self.COLOR_PALETTE_MODELS)
                    fig = g.fig
                    g.set_axis_labels(f"{atti}", "Counts")
                    g.set_xticklabels(rotation=90) 
                    fig.tight_layout()
                    figs[f"wattitude_{atti}_counts_ablation_{kgrp}"]=fig
            else:            
                fig = plt.figure(figsize=(12, 6))            
                sns.barplot(data= counts_for_plot(ds, var1="model", var2=f"{atti}_top1_label"), y="variable", x="value", hue="model" , palette=self.COLOR_PALETTE_MODELS)
                # plt.xticks(rotation=90)
                plt.xlabel(atti)
                plt.ylabel("Counts")
                fig.tight_layout()
                figs[f"wattitude_{atti}_counts"]=fig

            if ablation_vars:
                dati = self.counts_for_plot_ablation(ds, var1="model", var2=f"{atti}_top1_label", relative_to_var1_value="human")
                for kgrp, grp in dati.groupby(by=['ablation_group']):
                    g = sns.catplot(data=grp, x="value", y="variable", hue="model", col="ablation_variable", kind="bar", height=5, aspect=1.5, palette=self.COLOR_PALETTE_MODELS)
                    g.set_xticklabels(rotation=90) 
                    fig = g.fig
                    g.set_axis_labels(atti, "Counts relative to human")
                    fig.tight_layout()
                    figs[f"wattitude_{atti}_counts-relative2human_ablation_{kgrp}"]=fig
            else:        
                fig = plt.figure(figsize=(12, 6))
                sns.barplot(data=counts_for_plot(ds, var1="model", var2=f"{atti}_top1_label", relative_to_var1_value="human"), y="variable", x="value", hue="model", palette=self.COLOR_PALETTE_MODELS )
                # plt.xticks(rotation=90)
                plt.xlabel(atti)
                plt.ylabel("Counts relative to human")
                fig.tight_layout()
                figs[f"wattitude_{atti}_counts-relative2human"]=fig
            

            palette = sns.color_palette("husl")
            if ablation_vars:            
                dati = ds
                # for kgrp, grp in dati.groupby(by=['ablation_group']):
                for abvar in ablation_vars:
                    if abvar != "uncontrolled":
                        print(f"\n================= {abvar} ==============\n ")
                        for kgrp, grp in dati.query(f"ablation_variables=='uncontrolled' or ablation_variables=='{abvar}'").groupby(by=["model"]):
                            print(kgrp)
                            print(grp.emotion_top1_label.value_counts())
                            print("----------------------------\n")
                        g = sns.catplot(data=dati.query(f"ablation_variables=='uncontrolled' or ablation_variables=='{abvar}'"), 
                                        x=f"{atti}_top1_score", y="model", hue=f"{atti}_top1_label", kind="strip", height=5, aspect=1.5, palette=palette,
                                        col="ablation_variables", col_order=["uncontrolled", abvar])
                        g.set_xticklabels(rotation=90) 
                        fig = g.fig
                        # g.set_axis_labels(atti, "Counts relative to human")
                        fig.tight_layout()
                        figs[f"wattitude_{atti}_strip_ablation_uncontrolled-{abvar}"]=fig
            else:
                fig = plt.figure(figsize=(12, 6))                
                sns.stripplot(data=ds, y="model", x=f"{atti}_top1_score", hue=f"{atti}_top1_label", size=6, palette=palette)
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xlabel(f"Softmax score of the top-1 {atti}")            
                fig.tight_layout()
                figs[f"wattitude_{atti}_strip"]=fig
        #endregion

        return figs        
#endregion

import io

def main(log_wandb, localtest, ds_version, dswatti_version, t10sec, ds_artifact_name, **kwargs):
    tgt_dir = path.join(BASE_DIR, "out", "figs")
    logger.info(locals())

    tags=None
    if "ablation" in ds_artifact_name:
        kwargs["ablation_variables"] = "group"
        tags=["ablation"]

    run = None #wandb.init() # provide is art
    if log_wandb:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type=STEP_NAME, tags=tags) # provide is art

    # load dataset
    if localtest:
        ds_txts_fpath = f"{BASE_DIR}/artifacts/{ds_artifact_name}:{ds_version}/dataset.tsv"
        ds_writtersattitude_fpath = f"{BASE_DIR}/artifacts/writters_attitude:{dswatti_version}/merged_fmt.tsv"
    else:
        # use artifacts
        artifact, local_path = get_dataset(run, artifact_name=ds_artifact_name, ds_version=ds_version, **kwargs)
        ds_txts_fpath = path.join(local_path, artifact.files()[0].name)
        artifact, local_path = get_dataset(run, artifact_name="writters_attitude", ds_version=dswatti_version, **kwargs)
        ds_writtersattitude_fpath = path.join(local_path, "merged_fmt.tsv")

    logger.debug(ds_txts_fpath)
    logger.debug(ds_writtersattitude_fpath)
    dsan = DatasetAnalysis(ds_txts_fpath, ds_writtersattitude_fpath, **kwargs)

    figs = {}
    figs.update( dsan.plot_proportions_general() )
    figs.update( dsan.plot_proportions_attitude() )
    figs.update( dsan.plot_counts_wattitude(**kwargs) )

    # saving and logging plots
    for figname, fig in figs.items():
        logger.debug(f"Saving --> {figname}")        
        img_fpath = path.join(tgt_dir, f"{STEP_NAME}_{figname}.png")
        # wandb.log({f"{STEP_NAME}_{figname}".replace('_', '/'): plt}) # Not working
        # wandb.log({f"{STEP_NAME}_{figname}".replace('_', '/'): Image.open(img_fpath)}) # Not working
        # wandb.log({f"{STEP_NAME}_{figname}".replace('_', '/'): wandb.Image(fig)}) # Not working
        fig.savefig(img_fpath)  # using step_name as grouper 
        if log_wandb:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')        
            wandb.log({f"{STEP_NAME}_{figname}".replace('_', '/'): wandb.Image(Image.open(buf))})
    
    # plt.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(conflict_handler='resolve') # add_help=False to be used in cluster

    parser.add_argument("--ds_version", default="latest", type=str, help="dataset version")
    parser.add_argument("--dswatti_version", default="latest", type=str, help="Writters attitude version")
    parser.add_argument("--log_wandb", action="store_true", help="log to wandb")
    parser.add_argument("--localtest", action="store_true", help="ONLY use for local unit tests")    
    parser.add_argument('--t10sec', type=bool, default=False, help="Sanity check (Unitest)")
    parser.add_argument("--ds_artifact_name", default="ds_detection", type=str, help="name of the dataset to compute")

    args = parser.parse_args()

    main(**vars(args))
