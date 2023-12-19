import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import math
import numpy as np
from matplotlib import colors

def counts_for_plot(df:pd.DataFrame, var1, var2, relative_to_var1_value:str=None):
    dftmp = df[[var1, var2]].melt(id_vars=[var1])
    dftmp = dftmp.groupby(by=[var1, "value"]).count().reset_index()    
    if relative_to_var1_value:
        anchor_values = dict(dftmp[dftmp[var1]==relative_to_var1_value][["value", "variable"]].values)        
        dftmp["variable"] = dftmp.apply(lambda e: e["variable"] - anchor_values.get(e["value"], 0), axis=1)
        dftmp = ( dftmp.query(f"{var1}!='{relative_to_var1_value}'") ) # remove anchor entries
    return dftmp

def proportion_plot(labels, counts, **kwargs):    
    palette_color = sns.color_palette('muted') 
    plt.pie(counts, labels=labels, colors=palette_color, autopct='%.0f%%')  

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def define_color_palette(ds, for_column="model", base_palette=sns.color_palette(), fixed={'human': '#7f7f7f'}): #1f77b4
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


def make_chart_radar(df, series_column=None, color_or_color_map=None, fig:plt.Figure=None, subplot_specs=None, ylim=None, yticks=None, plot_legend=True):
    
    rs = {}
    categories=[c for c in df.columns if c != series_column]
    N = len(categories)

    
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
     
    if not fig:
        # plt.rc('figure', figsize=(12, 12))
        fig = plt.figure(figsize=(12, 12))

    if subplot_specs:        
        ax = fig.add_subplot(subplot_specs, polar=True)
    else:
        ax = plt.subplot(1,1,1, polar=True)    
    
    rs["ax"] = ax

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
 
    plt.xticks(angles[:-1], categories, color='black', size=14)#12)
    # ax.tick_params(axis='x', rotation=5.5)
    ax.tick_params(axis='x')
    
    ax.set_rlabel_position(0)
    # plt.yticks([.20,.40,.60,.80], [".20",".40",".60",".80"], color="black", size=10)
    # plt.ylim(0,1.00)
    if yticks:
        plt.yticks(*yticks, color="black", size=14)#10)
    if ylim:
        plt.ylim(*ylim)
 
    
    if not color_or_color_map:
        color_or_color_map = {} if series_column else "blue"
    
    
    _label = None
    _color = color_or_color_map
    for row in range(0, len(df.index)):
        if series_column:
            _values = df.iloc[row].drop([series_column]).values.flatten().tolist()
            _label = df.iloc[row][series_column]
            _color = color_or_color_map.get(_label, None)
        else:
            _values = df.iloc[row].values.flatten().tolist()

        _values+= _values[:1]
        ax.plot(angles, _values, 'o-', linewidth=2, label = _label, color=_color)#, linestyle='solid')
        ax.fill(angles, _values, alpha=0.2, color=_color)

    if plot_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, shadow=False, ncol=1, frameon=False, fontsize=14)#10)
    else:
        handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        rs.update({"legend_handles": handles, "legend_labels": labels})
        
    return rs

def plot_correlation_matrix(df_corrs, significance_level=0.05, fig=None, cbar=True) -> plt.Figure:    
    tmp = np.ma.masked_where(df_corrs>significance_level, df_corrs)
    if not fig:
        fig = plt.figure(figsize=(10, 8))
    g = sns.heatmap(df_corrs, cmap='viridis', annot=True, cbar=cbar )
    sns.heatmap(df_corrs, cmap=colors.ListedColormap(['lightgray']), mask=(tmp>significance_level), cbar=False, annot=True )
    # plt.subplots_adjust(left=0.2)#, hspace=0.1)
    # plt.savefig("../out/figs/correlations_categorical.png")
    # plt.show()
    return fig