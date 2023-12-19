"""Evaluate performance of detectors

* Compute metrics
* Generate related charts

Metrics computed:
    rs.update({"tn": tn, "fp":fp, "fn":fn, "tp":tp})
    rs["tpr"] = tp / (tp + fn) # sensitivity
    rs["tnr"] = tn / (tn + fp) # specificity
    rs["fpr"] = fp / (fp + tn) # fall-out
    rs["fnr"] = fn / (fn + tp) # miss-rate

    rs["samples"] = len(y_pred)
    rs["support"] = sum(y_true)
    rs["precision"] = tp/(tp+fp)
    rs["error_rate"] = (fp+fn)/(tp+fp+tn+fn) # ->0
    rs["khat"]= (2*(tp*tn-fn*fp)) / ((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)) # ->1 Cohen’s Kappa Coefficient (normalizes the accuracy by the possibility of agreement by chance)
    rs["mcc"]= (tp*tn-fp*fn) / math.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) ) # ->1 Matthews Correlation Coefficient (good for unbalanced cases)
    rs["csi"]= tp/(tp+fp+tn) # ->1 Critical Success Index (aka threat score (TS) or Jaccard’s Index)

    rs["gmean"] = math.sqrt(rs["recall"]+rs["tnr"]) # -> Geometric mean (The higher the value the lower the risk of over-fitting of negative and under-fitting of positive classes)
    rs["balac"] = (rs["recall"]+rs["tnr"])/2 # Balanced accuracy (useful when the number of observations across classes is imbalanced)
    rs["posLr"] = rs["tpr"]/rs["fpr"] # ->1 Positive Likelihood Ratio -> odds of obtaining a positive prediction for actual positives
    rs["negLr"] = rs["fnr"]/rs["tnr"] # ->0 Negative Likelihood Ratio -> odds of obtaining a negative prediction for actual positives 
    rs["dor"] = rs["posLr"]/rs["negLr"] # -> Diagnostic Odds Ratio -> metric summarizing the effectiveness of classification
    rs["bmi"] = rs["tpr"] + rs["tnr"] - 1 # ->1 Informedness


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
from .utils import get_dataset, get_step_config
from .utils_plots import define_color_palette, make_chart_radar

import warnings
from sklearn.exceptions import UndefinedMetricWarning

import settings
BASE_DIR=settings.BASE_DIR
WANDB_ENTITY=settings.WANDB_ENTITY
WANDB_PROJECT=settings.WANDB_PROJECT
CONFIG_PIPELINE=settings.CONFIG_PIPELINE
STEP_NAME="evaluate-detectors"
STEP_CONFIG=get_step_config(STEP_NAME)

logger = settings.logging.getLogger(__name__)
settings.logging.getLogger('matplotlib').setLevel(level='INFO') # too much DEBUG msgs from plt
settings.logging.getLogger('urllib3').setLevel(level='INFO') # too much DEBUG msgs from it
#region local_libs ========================================


#endregion

#region main_code =========================================================
class EvaluateDetectors( ):

    COLOR_PALETTE_MODELS = []
    ds = None
    metric_titles = {"tpr": "Sensitivity", "tnr": "Specificity", "fpr": "Fall-out rate", "fnr": "miss rate", 
                "khat": "Cohen's Kappa Coefficient", "mcc": "Matthews Correlation Coefficient", "csi": "Critical Success Index",
                "gmean": "Geometric mean", "balac": "Balanced accuracy", 
                "posLr" : "Positive Likelihood ratio", "negLr" : "Negative Likelihood ratio", 
                "dor" : "Diagnostic Odds Ratio", "bmi": "Informedness", 
                "f1": "F1", "f1-weighted": "F1-weighted", "f1-macro": "F1-macro", 
                "error_rate": "Error rate", "precision": "Precision", "recall": "Recall"
                }
    metrics_tend_to_1 = ["accuracy", "balac", "precision", "recall", "tpr", "tnr", "f1", "f1-weighted", "f1-macro", "mcc", "khat", "roc_auc_ovr", "csi", "bmi"]
    sklearn_metrics=["accuracy", "recall", "roc_auc_ovr", "f1",                      
                     {'metric': 'f1', 'name': 'f1-weighted', 'params': {'average': 'weighted'}}, 
                     {'metric': 'f1', 'name': 'f1-macro', 'params': {'average': 'macro'}}, 
                     ]
    
    def __init__(self, ds_txts_fpath:str, ds_writtersattitude_fpath:str, **kwargs) -> None:
        print(locals())
        self.ds = self._preprocess(self.get_ds(ds_txts_fpath, ds_writtersattitude_fpath), cols2keep=kwargs.get("cols2keep", []))
        print(f"STEP_CONFIG --> {STEP_CONFIG}")

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

        print(ds.shape)
        print(ds.columns.tolist())

        return ds
    
    def _preprocess(self, df, cols2keep=[]):
        # select cols of interest
        # df = df[["data_id", "id", "category", "model"] + [c for c in df.columns if c.startswith("detector_") and c.endswith("_label")]].copy()
        df = df[["data_id", "id", "category", "model"] + cols2keep  + [c for c in df.columns if c.endswith("_label")]].copy() # now use all labels
        
        # homogenize labels
        df["model_det"] = df["model"]
        label_map = {
            'model': {'human': "Human", '*': "LLM"},
            'detector_radar_vicuna_7B_label': {'AI-generated': "LLM", 'Human-written': "Human"},
            'detector_chatgpt_qa_detector_label': {'LABEL_1': "LLM", 'LABEL_0': "Human"},
            'detector_xlmr_chatgptdetect_noisy_label': {'CHATGPT': "LLM", 'HUMAN': "Human"},
            'detector_llm_det_label': {'Human_write': "Human", '*': "LLM"}
        }


        for c in df.columns:
            if c in label_map.keys():
                mapping = label_map[c]
                if '*' in mapping: # expand wildcard
                    mapping = {**{v: mapping['*'] for v in df[c].unique().tolist()}, **mapping}            
                # df[c] = df[c].map(mapping)
                df[c] = df[c].replace(mapping)

        return df

#region evaluation
    def evaluate(self, y_true, y_pred, scorers):
        rs = {}
        for scorer in scorers:
            try:
                if isinstance(scorer, dict):
                    # print(f"-->{scorer}")
                    _metric, _name, _params = scorer.get("metric"), scorer.get("name"), scorer.get("params")
                    # print(f"{_metric}, {_name}, {_params}")
                else:
                    _metric, _name, _params = scorer, scorer, {}
                scoring_func = metrics.get_scorer(_metric) #scorer)
                sc = scoring_func._score_func(y_true, y_pred, **_params)
                # rs[scorer] = sc
                rs[_name] = sc
            except Exception as err:
                print(err)
                rs[scorer] = None
                pass            

        # tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() # this is the right order of labels for Positive = 1 Negative = 0
        rs.update({"tn": tn, "fp":fp, "fn":fn, "tp":tp})
        # https://cran.r-project.org/web/packages/metrica/vignettes/available_metrics_classification.html
        rs["tpr"] = tp / (tp + fn) # sensitivity -> how well the positive class was predicted
        rs["tnr"] = tn / (tn + fp) # specificity -> how well the negative class was predicted
        rs["fpr"] = fp / (fp + tn) # fall-out
        rs["fnr"] = fn / (fn + tp) # miss-rate

        rs["samples"] = len(y_pred)
        rs["support"] = sum(y_true)
        rs["precision"] = tp/(tp+fp)
        rs["recall"] = tp/(tp+fn)
        rs["error_rate"] = (fp+fn)/(tp+fp+tn+fn) # ->0
        rs["khat"]= (2*(tp*tn-fn*fp)) / ((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)) # ->1 Cohen’s Kappa Coefficient (normalizes the accuracy by the possibility of agreement by chance)
        rs["mcc"]= (tp*tn-fp*fn) / math.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) ) # ->1 Matthews Correlation Coefficient (good for unbalanced cases)
        rs["csi"]= tp/(tp+fp+tn) # ->1 Critical Success Index (aka threat score (TS) or Jaccard’s Index)

        rs["gmean"] = math.sqrt(rs["tpr"] * rs["tnr"]) # math.sqrt(rs["recall"]+rs["tnr"]) # -> Geometric mean (The higher the value the lower the risk of over-fitting of negative and under-fitting of positive classes)
        rs["balac"] = (rs["recall"]+rs["tnr"])/2 # Balanced accuracy (useful when the number of observations across classes is imbalanced)
        rs["posLr"] = rs["tpr"]/rs["fpr"] # ->1 Positive Likelihood Ratio -> odds of obtaining a positive prediction for actual positives
        rs["negLr"] = rs["fnr"]/rs["tnr"] # ->0 Negative Likelihood Ratio -> odds of obtaining a negative prediction for actual positives 
        rs["dor"] = rs["posLr"]/rs["negLr"] # -> Diagnostic Odds Ratio -> metric summarizing the effectiveness of classification
        rs["bmi"] = rs["tpr"] + rs["tnr"] - 1 # ->1 Informedness
        # rs["auc_roc"] = (tp/(tp+fp))+(tp/(tp+fp))-1  # ppv_or_precision+npv−1

        return rs

    def evaluate_detector(self, df, prediction_column, true_column="model", 
                        #   scorers=["accuracy", "recall", "f1", {'metric': 'f1', 'name': 'f1-weighted', 'params': {'average': 'weighted'}}, "roc_auc_ovr"], 
                          scorers=sklearn_metrics, 
                          positive_label=STEP_CONFIG.get('positive_label', "Human")):
        dftmp = df[[true_column, prediction_column]]
        dftmp = dftmp.dropna() # filter out error predictions 
        dftmp[true_column] = dftmp[true_column].map(lambda v: 1 if v==positive_label else 0)
        dftmp[prediction_column] = dftmp[prediction_column].map(lambda v: 1 if v==positive_label else 0)    

        return self.evaluate(dftmp[true_column].values, dftmp[prediction_column].values, scorers)
    
    def run_evaluations(self, df=None, groups=["category"], subgroups=[]):
        """run evaluations for all the detectors and all the task-detector combinations

        Args:
            df (_type_): _description_
            groups (list, optional): _description_. Defaults to ["category"].

        Returns:
            _type_: _description_
        """
        if df is None:
            df = self.ds
        tasks = list(df.category.unique())
        rs = []
        for detector_column in [c for c in df.columns if c.startswith("detector_")]:
            _detector_name = detector_column[len("detector_"):]
            rs.append({**{"detector": _detector_name, "generator": None, "group": None}, 
                    **self.evaluate_detector(df, prediction_column=detector_column)})
            for grp in groups:
                tasks = list(df.category.unique())
                # print(df[grp].unique().tolist)
                for gv in df[grp].unique():
                    rs.append({**{"detector": _detector_name, "generator": None, "group": f"{grp}:{gv}"}, 
                            **self.evaluate_detector(df.query(f"{grp}=='{gv}'"), prediction_column=detector_column)})
                    for sgrp in subgroups:
                        for sgv in df[sgrp].unique():
                            rs.append({**{"detector": _detector_name, "generator": None, "group": f"{grp}:{gv}|{sgrp}:{sgv}"}, 
                                    **self.evaluate_detector(df.query(f"{grp}=='{gv}' and {sgrp}=='{sgv}'"), prediction_column=detector_column)})                    
        
        return rs
    
    def run_evaluations_by_generator(self, df=None, groups=["category"], generator_col="model_det"):
        # run evaluations per detector and nested by generator and by group/
        # on generator grouping includes the human ones to have comparisons
        if df is None:
            df = self.ds

        rs = []
        for detector_column in [c for c in df.columns if c.startswith("detector_")]:
            _detector_name = detector_column[len("detector_"):]
            # rs.append({**{"detector": _detector_name, "generator": None, "group": None}, 
            #         **self.evaluate_detector(df, prediction_column=detector_column)})
            
            generator_col = "model_det"
            for gen in df[generator_col].unique():
                if gen!='human' and gen:
                    rs.append({**{"detector": _detector_name, "generator": f"{gen}", "group": None},
                                            **self.evaluate_detector(df.query(f"{generator_col}=='{gen}' or {generator_col}=='human'"), prediction_column=detector_column)})            

                    for grp in groups:                    
                        for gv in df[grp].unique():
                            rs.append({**{"detector": _detector_name, "generator": f"{gen}", "group": f"{grp}:{gv}"}, 
                                    **self.evaluate_detector(df.query(f"({generator_col}=='{gen}' or {generator_col}=='human') and {grp}=='{gv}'"), prediction_column=detector_column)})
        
        return rs    
#endregion

#region plots
    def plot_overview_general_bymetric(self, df, metrics=metrics_tend_to_1, individual=True):
        rs = {}
        # create plots area
        fig = plt.figure(figsize=(25, 10))
        gs = plt.GridSpec(nrows=1, ncols=2, width_ratios=[1,1], wspace=0.1, hspace=0)

        dftmp = df[(df.generator.isna()) & (df.group.isna())][ ["detector"] + metrics ]
        dftmp.rename(self.metric_titles, axis=1, inplace=True)
        # display(dftmp.head())
        # if individual:
        #     make_chart_radar(dftmp, series_column="detector", ylim=(0.0,1.0), color_or_color_map=self.COLOR_PALETTE_MODELS, 
        #                     fig=fig)
        # else:
        make_chart_radar(dftmp, series_column="detector", ylim=(0.0,1.0), color_or_color_map=self.COLOR_PALETTE_MODELS, 
                        fig=fig, subplot_specs=None if individual else gs[0,0], plot_legend=individual)            
        # plt.title("General performance (on metrics -> 1)", fontsize=16, x = 0.5, y = 1.05)
        if individual:
            fig.suptitle('General performance by metric', fontsize=16)
            fig.tight_layout()            
            rs.update({f"overview_general_bymetric{'-radar'}": fig})

        if individual:
            fig = plt.figure(figsize=(14, 7))
            pass
        else:
            ax = fig.add_subplot(gs[0, 1])
        dftmp = df[(df.generator.isna()) & (df.group.isna())].head()[["detector"]+["gmean", "error_rate", "posLr", "negLr", "dor"]].melt(id_vars=["detector"])
        # dftmp.rename({"variable":"metric"}, axis=1, inplace=True)
        dftmp.variable = dftmp.variable.map(self.metric_titles)
        # display(dftmp)
        sns.barplot(data=dftmp, x="variable", y="value", hue="detector", palette=self.COLOR_PALETTE_MODELS)
        # plt.xticks(rotation=45)
        plt.xlabel("Metrics")
        # plt.ylabel("Counts")
        fig.suptitle('General performance by metric', fontsize=16)
        fig.tight_layout()        
        rs.update({f"overview_general_bymetric{'-bar' if individual else ''}": fig})

        # fig2 = plt.figure(figsize=(12, 12))
        # sns.barplot(data=dftmp, x="variable", y="value", hue="detector", palette=self.COLOR_PALETTE_MODELS)        
        # rs.update({f"overview_general_bymetric-bar": fig2})

        # return {f"overview_general_bymetric": fig}
        return rs

    
    def plot_radars_bytask_bymetric(self, df, metrics=metrics_tend_to_1, title="Performance by task"):
        df_filtered = df[(df.generator.isna()) & (~df.group.isna()) & (df.group.str.contains('\|')==False)] # only results without subgroups

        # create plots area
        fig = plt.figure(figsize=(25, 20))
        n_figs = len(df_filtered.group.unique())
        n_cols = 3
        gs = plt.GridSpec(nrows=math.ceil(n_figs/n_cols), ncols=n_cols, width_ratios=[1]*n_cols, wspace=0.0, hspace=0.4)

        for i, (grp, dfi) in enumerate(df_filtered.groupby(by=["group"])):            
            dftmp = dfi[ ["detector"] + metrics ].copy()
            dftmp.rename(self.metric_titles, axis=1, inplace=True)
            chart_rs = make_chart_radar(dftmp, series_column="detector", ylim=(0.0,1.0), color_or_color_map=self.COLOR_PALETTE_MODELS, 
                            fig=fig, subplot_specs=gs[int(i/n_cols), (i%n_cols)], plot_legend= False )#((i+1)==n_figs))
            chart_rs['ax'].set_title(grp.split(":")[1], fontsize=15)
            
        fig.legend(chart_rs['legend_handles'], chart_rs['legend_labels'], bbox_to_anchor=(0.8, 0.2))#, loc='lower right')

        fig.suptitle(title, fontsize=16)
        # plt.subplots_adjust(top=0.85)  
        fig.tight_layout()
        return {f"radars_bytask_bymetric": fig}


    def plot_overview_bytask_onemetric(self, df, metric="f1-weighted", individual=False, title="Overview by task"):

        rs ={}
        df_filtered = df[(df.generator.isna()) & (~df.group.isna()) & (df.group.str.contains('\|')==False)] # only results without subgroups
        dftmp = df_filtered[~df_filtered.group.isna()][["detector", "group", metric]].pivot_table(index="detector", columns="group", values=metric).reset_index()
        # dftmp.rename({c:c.split(":")[1] for c in dftmp.columns if c.startswith("category:")}, axis=1, inplace=True)
        dftmp.rename({c:c.split(":")[1] for c in dftmp.columns if (":" in c)}, axis=1, inplace=True)
        # display(dftmp.head())

        # create plots area
        fig = plt.figure(figsize=(12, 10)) if individual else plt.figure(figsize=(25, 12))
        gs = plt.GridSpec(nrows=1, ncols=2, width_ratios=[1,1], wspace=0.1, hspace=0)

        make_chart_radar(dftmp, series_column="detector", ylim=(0.0,1.0), color_or_color_map=self.COLOR_PALETTE_MODELS, fig=fig, 
                         subplot_specs=None if individual else gs[0,0])
        # plt.title("F1-score by task", fontsize=16, x = 0.5, y = 1.1)
        if individual:
            fig.suptitle(f'{title} ({self.metric_titles.get(metric, metric)})', fontsize=16)
            fig.tight_layout()            
            rs.update({f"overview_bytask-{metric}{'-radar'}": fig})

        if individual:
            fig = plt.figure(figsize=(14, 7))
            pass
        else:
            ax = fig.add_subplot(gs[0, 1])
        sns.barplot(data=df_filtered, x="detector", y=metric, hue="group")
        plt.xticks(rotation=45)
        plt.xlabel("Detectors")
        for lg in plt.legend().get_texts():
            lg.set_text(lg.get_text().split(":")[1])
        # plt.ylabel("Counts")

        fig.suptitle(f'{title} ({self.metric_titles.get(metric, metric)})', fontsize=16)
        plt.subplots_adjust(bottom=0.15) #to avoid cuttingoff rotatted bottom labels
        fig.tight_layout()
        # plt.show()
        rs.update({f"overview_bytask-{metric}{'-bar' if individual else ''}": fig})
        # return {f"overview_bytask-{metric}": fig}
        return rs
    

    def plot_detector_vs_generator_overview(self, df, metric="mcc", title="Detector vs Generator"):

        dftmp = df[(~df.generator.isna()) & (df.group.isna())][ ["detector", "generator", "group"] + [metric] ]
        dftmp.generator = dftmp.generator.map(lambda v: v.split('/')[-1])
        dftmp = dftmp.pivot(index=["detector"], columns=["generator"], values=metric).reset_index()

        fig = plt.figure(figsize=(12, 10))
        make_chart_radar(dftmp, series_column="detector", ylim=(0.0,1.0), color_or_color_map=self.COLOR_PALETTE_MODELS, 
                            fig=fig)#, subplot_specs=gs[0,0],  plot_legend=True)

        fig.suptitle(f'{title} ({self.metric_titles.get(metric, metric)})', fontsize=16)
        fig.tight_layout()
        return {f"detectorVSgenerator_overview-{metric}": fig}

    def plot_detector_vs_generator_bytask(self, df, metrics=["mcc", "gmean", "f1-weighted"], title="Detector vs Generator"):

        dftmp = df[~df.generator.isna()][ ["detector", "generator", "group"] + metrics ]
        dftmp.group = dftmp.group.map(lambda v: v.split(':')[-1] if v else None)
        dftmp = dftmp.melt(id_vars=["detector", "generator", "group"], value_vars=metrics)
        dftmp.rename({'group': 'task', 'variable': 'metric'}, inplace=True, axis=1)
        g = sns.catplot(
            data=dftmp, kind="bar",
            x="generator", y="value", col="metric", row="task", hue="detector",
            height=10, aspect=1.4,
        )
        g.set_xticklabels(rotation=90) 
        g.fig.tight_layout()
        return {f"detectorVSgenerator_bytask-{len(metrics)}": g.fig}
#endregion   

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
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type=STEP_NAME, tags=tags) # provide is art

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
    evaluator = EvaluateDetectors(ds_fpath, ds_writtersattitude_fpath, cols2keep=cols2keep, **kwargs)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning) # to avoid warnings of metrics when no enough data of all classes
        warnings.simplefilter("ignore", category=RuntimeWarning)
        np.seterr(invalid='ignore') # Suppress/hide the warning (e.g. True divide)
        # rs = evaluator.run_evaluations(groups=["category"], subgroups=["emotion_top1_label", "irony_top1_label", "convincingness_top1_label", "persuasiveness_top1_label"])   
        df_rs = pd.concat([pd.DataFrame.from_dict(evaluator.run_evaluations(groups=groups)), 
                           pd.DataFrame.from_dict(evaluator.run_evaluations_by_generator(groups=groups))])


    detector_rename = {
        'detector_chatgpt_qa_detector': 'ChatGPT_QA',
        'detector_xlmr_chatgptdetect_noisy': 'XLMR_ChatGPT',
        'detector_llm_det': 'LLMDet',
        'detector_gpt_zero': 'GPT_Zero',
        'detector_radar_vicuna_7B': 'Radar_Vicuna7B'
    }
    detector_rename = {f"{k.replace('detector_', '')}_label":v for k,v in detector_rename.items()}
    df_rs.detector = df_rs.detector.map(detector_rename)

    # df_rs = pd.DataFrame.from_dict(rs)
    rs_fpath = path.join(tgt_dir_files, f"detectors_evaluation_metrics{suffix}.tsv")
    df_rs.to_csv(rs_fpath, sep='\t')

    if log_wandb:
        artifact_rs = wandb.Artifact(name=STEP_NAME, type="dataset")
        artifact_rs.add_file(local_path=rs_fpath, name="detectors_evaluation_metrics.tsv")
        run.log_artifact(artifact_rs)

    # generate plots 
    evaluator.COLOR_PALETTE_MODELS = define_color_palette(df_rs, for_column="detector", fixed={'human': '#7f7f7f'}) # human -> gray # fixed color-palette
    figs = {}
    figs.update( evaluator.plot_overview_general_bymetric(df_rs) )
    figs.update( evaluator.plot_overview_general_bymetric(df_rs, individual=True) )
    figs.update( evaluator.plot_radars_bytask_bymetric(df_rs, title="Ablation performance by variable" if tags else "Performance by task") )
    figs.update( evaluator.plot_overview_bytask_onemetric(df_rs, metric='f1-macro', title="Ablation by variable" if tags else "Overview by task") )
    figs.update( evaluator.plot_overview_bytask_onemetric(df_rs, metric='f1-macro', individual=True, title="Ablation by variable" if tags else "Overview by task") )
    figs.update( evaluator.plot_detector_vs_generator_overview(df_rs, metric="mcc"))
    figs.update( evaluator.plot_detector_vs_generator_overview(df_rs, metric="f1-macro"))
    figs.update( evaluator.plot_detector_vs_generator_bytask(df_rs, metrics=["mcc", "gmean", "f1-macro"]))
    figs.update( evaluator.plot_detector_vs_generator_bytask(df_rs, metrics=["f1-macro"]))

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
