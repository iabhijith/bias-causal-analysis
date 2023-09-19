import json
import yaml
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import polars as pl
import plotly.io as pio

from pathlib import Path
from plotly.subplots import make_subplots

RESULTS = "results"


def top_n_heads(arr, n):
    flat_arr = arr.flatten() 
    flat_arr.sort()           
    threshold = flat_arr[-n]
    result = np.where(arr >= threshold, 1, 0)
    return result.tolist()

def plot_heads(arr, top_n, title):
    fig = px.imshow(arr, color_continuous_scale='Blues', origin='lower', title=title)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_cma_heads(filepath, top_n):
    result = json.load(filepath.open())
    arr = result['indirect_effects']
    arr = np.array(arr)
    arr = top_n_heads(arr, top_n)
    return plot_heads(arr, top_n, "Top {} heads using CMA".format(top_n)), arr

def plot_diffmask_heads(filepath, top_n):
    result = json.load(filepath.open())
    arr = result['mask']
    arr = np.array(arr)
    arr = top_n_heads(arr, top_n)
    return plot_heads(arr, top_n, "Top {} heads using DiffMask".format(top_n)), arr

def plot_acdc_heads(filepath, top_n):
    with open(filepath, 'r') as f:
        components = yaml.safe_load(f)
    heads = components['attn_heads']['subset']
    arr = np.zeros((12, 12))
    for l in heads.keys():
        for h in heads[l]:
            arr[int(l), int(h)] = 1
    arr = np.array(arr)
    arr = top_n_heads(arr, top_n)
    return plot_heads(arr, top_n, "Top {} heads using ACDC".format(top_n)), arr

def box_plot(df, figure_path, score_column, y_label):
    box_df = df.filter(pl.col("model").is_in(["gpt2_baseline"]).is_not())
    box_df = box_df.sort([pl.col("model_type"), pl.col("model_order")]).to_pandas()
    fig = px.box(box_df, color='model_type', color_discrete_sequence=px.colors.qualitative.Plotly,
                 y=score_column, x='model_label', points='all')
    fig.write_image(figure_path, format="pdf")
    time.sleep(2)
    fig.update_xaxes(title='', visible=True, showticklabels=True, tickangle=0)
    fig.update_yaxes(title=y_label, visible=True, showticklabels=True, automargin=True)
    fig.update_layout(
        autosize=True,
        font_family="Computer Modern",
        margin=dict(l=2, r=0, b=0, t=0, pad=0),
        font=dict(size=25),
        showlegend=False,
        # xaxis={'tickwidth': 10},
    )
    fig.update_traces(width=0.5)
    fig.layout.xaxis2 = go.layout.XAxis(overlaying='x', range=[0, 2], showticklabels=False)

    baseline = df.filter(pl.col("model").is_in(["gpt2_baseline"])).to_pandas()
    baseline = baseline[score_column].mean()
    fig.add_scatter(x=[0, 2], y=[baseline, baseline], mode='lines', xaxis='x2',
                    showlegend=False, line=dict(dash='dash', color='#8E44AD'))

    fig.write_image(figure_path, format="pdf", width=1000, height=1000)

def plot_results():
    filter = [] #["gpt2_attn_layers","gpt2_attn_layers_all"]
    labels = {
        "gpt2_baseline": "gpt2",
        "gpt2_all": "full<br>model",
        "gpt2_random": "random<br>attn<br>heads",
        "gpt2_acdc": "acdc",
        "gpt2_attn_heads_cma_top10": "cma<br>attn<br>heads",
        "gpt2_attn_heads_dm_top10": "dm<br>attn<br>heads",
        "gpt2_attn_heads_acdc": "acdc<br>attn<br>heads",
        "gpt2_attn_layers": "last 4<br>attn<br>layers",
        "gpt2_attn_layers_all": "all<br>attn<br>layers",
    }
    baselines  = {
        "gpt2_baseline": "0baseline",
        "gpt2_all": "0baseline",
        "gpt2_random": "0baseline",
        "gpt2_acdc": "1layers",
        "gpt2_attn_heads_cma_top10": "2heads",
        "gpt2_attn_heads_dm_top10": "2heads",
        "gpt2_attn_heads_acdc": "2heads",
        "gpt2_attn_layers": "1layers",
        "gpt2_attn_layers_all": "1layers",
    }

    order = {
        "gpt2_baseline": "0",
        "gpt2_all": "1",
        "gpt2_random": "2",
        "gpt2_acdc": "5",
        "gpt2_attn_heads_cma_top10": "7",
        "gpt2_attn_heads_dm_top10": "8",
        "gpt2_attn_heads_acdc": "6",
        "gpt2_attn_layers": "4",
        "gpt2_attn_layers_all": "3",
    }


    results_path = Path("results/evaluation")
    list_of_files = list(results_path.glob('*.json'))
    results = []
    for f in list_of_files:
        results.append(pl.read_ndjson(f))
    results = pl.concat(results)
    results = results.filter(pl.col("model").is_in(filter).is_not())
    results = results.with_columns([
       pl.col("model").apply(lambda x: baselines[x]).alias("model_type"),
        pl.col("model").apply(lambda x: labels[x]).alias("model_label"),
        pl.col("model").apply(lambda x: order[x]).alias("model_order"),
    ])
    
    box_plot(results, "results/plots/crows_pairs.pdf", "CrowS-Pairs", "CrowS-Pairs Stereotype Score")
    box_plot(results, "results/plots/professions.pdf", "Professions", "Professions Stereotype Score")

        
    # fig = px.box(results.to_pandas(), color="model", y="CrowS-Pairs", points='all', title="CrowS-Pairs results")
    # pio.write_image(fig, "results/plots/crows_pairs.png", format='png', width=1000, height=1000)
    
    results_stereoset = results.select(["model", 'seed',"model_type", "model_label","model_order", "LM Score", "SS Score", "ICAT"])
    # fig = px.box(results_stereoset, color="model", y="LM Score", points='all', title="SteroeSet: LM Score")
    # pio.write_image(fig, "results/plots/ss_lmscore.png", format='png', width=1000, height=1000)
    box_plot(results_stereoset, "results/plots/ss_lmscore.pdf", "LM Score", "StereoSet LM Score")

    # fig = px.box(results_stereoset, color="model", y="SS Score", points='all', title="SteroeSet: SS Score")
    # pio.write_image(fig, "results/plots/ss_ssscore.png", format='png', width=1000, height=1000)
    box_plot(results_stereoset, "results/plots/ss_ssscore.pdf", "SS Score", "StereoSet SS Score")

    # fig = px.box(results_stereoset, color="model", y="ICAT", points='all', title="SteroeSet: ICAT")
    # pio.write_image(fig, "results/plots/ss_icat.png", format='png', width=1000, height=1000)
    box_plot(results_stereoset, "results/plots/ss_icat.pdf", "ICAT", "StereoSet ICAT")
    
    
    results_blimp = results.select(["model", "seed", "model_type", "model_label","model_order", "BLiMP", "BLiMP AGA", "BLiMP ISV1", "BLiMP ISV2", "BLiMP RSV1", "BLiMP RSV2"])
    results_blimp = results_blimp.with_columns([
        ((pl.col("BLiMP ISV1") + pl.col("BLiMP RSV1") + pl.col("BLiMP ISV2") + pl.col("BLiMP RSV2"))/4).alias("BLiMP SV")])
    results_blimp = results_blimp.select(["model", "seed","model_type", "model_label","model_order", "BLiMP", "BLiMP AGA", "BLiMP SV"])
    #results_blimp = pd.melt(results_blimp.to_pandas(), id_vars=['model', 'seed'], value_vars=['BLiMP', 'BLiMP AGA', 'BLiMP SV']).rename(columns={'variable': 'metric', 'value': 'score'})

    # fig = px.box(results_blimp, color="model", y="BLiMP", points='all', title="BLiMP results")
    # pio.write_image(fig, "results/plots/blimp.png", format='png', width=1000, height=1000)

    box_plot(results_blimp, "results/plots/blimp.pdf", "BLiMP", "BLiMP Overall Score")

    # fig = px.box(results_blimp, color="model", y="BLiMP AGA", points='all', title="BLiMP AGA results")
    # pio.write_image(fig, "results/plots/blimp_aga.png", format='png', width=1000, height=1000)

    box_plot(results_blimp, "results/plots/blimp_aga.pdf", "BLiMP AGA", "BLiMP AGA Score")

    # fig = px.box(results_blimp, color="model", y="BLiMP SV", points='all', title="BLiMP SV results")
    # pio.write_image(fig, "results/plots/blimp_sv.png", format='png', width=1000, height=1000)

    box_plot(results_blimp, "results/plots/blimp_sv.pdf", "BLiMP SV", "BLiMP SV Score")


    results_winobias = results.select(["model", "seed", "model_type", "model_label", "model_order", "WinoBias Type1 Dev", "WinoBias Type1 Test", "WinoBias Type2 Dev", "WinoBias Type2 Test"])
    results_winobias = results_winobias.with_columns([
        ((pl.col("WinoBias Type1 Dev") + pl.col("WinoBias Type1 Test"))/2).alias("WinoBias1"),
        ((pl.col("WinoBias Type2 Dev") + pl.col("WinoBias Type2 Test"))/2).alias("WinoBias2"),
        ((pl.col("WinoBias Type1 Dev") + pl.col("WinoBias Type1 Test") + pl.col("WinoBias Type2 Dev") + pl.col("WinoBias Type2 Test"))/4).alias("WinoBias")
        ])
    results_winobias = results_winobias.select(["model", "seed", "model_type", "model_label","model_order", "WinoBias", "WinoBias1", "WinoBias2"])

    # fig = px.box(results_winobias.to_pandas(), color="model", y="WinoBias", points='all', title="WinoBias Overall results")
    # pio.write_image(fig, "results/plots/winobias.png", format='png', width=1000, height=1000)
    box_plot(results_winobias, "results/plots/winobias.pdf", "WinoBias", "WinoBias Overall Score")

    # fig = px.box(results_winobias.to_pandas(), color="model", y="WinoBias1", points='all', title="WinoBias Type1 results")
    # pio.write_image(fig, "results/plots/winobias1.png", format='png', width=1000, height=1000)
    box_plot(results_winobias, "results/plots/winobias1.pdf", "WinoBias1", "WinoBias Type1 Score")
    # fig = px.box(results_winobias.to_pandas(), color="model", y="WinoBias2", points='all', title="WinoBias Type2 results")
    # pio.write_image(fig, "results/plots/winobias2.png", format='png', width=1000, height=1000)
    box_plot(results_winobias, "results/plots/winobias2.pdf", "WinoBias2", "WinoBias Type2 Score")

    return results

def get_results():
    filter = [] #["gpt2_attn_layers","gpt2_attn_layers_all"]
    labels = {
        "gpt2_baseline": "gpt2",
        "gpt2_all": "full model",
        "gpt2_random": "random attn heads",
        "gpt2_acdc": "acdc",
        "gpt2_attn_heads_cma_top10": "cma attn heads",
        "gpt2_attn_heads_dm_top10": "dm attn heads",
        "gpt2_attn_heads_acdc": "acdc attn heads",
        "gpt2_attn_layers": "last 4 attn layers",
        "gpt2_attn_layers_all": "all attn layers",
    }
    baselines  = {
        "gpt2_baseline": "0baseline",
        "gpt2_all": "0baseline",
        "gpt2_random": "0baseline",
        "gpt2_acdc": "1layers",
        "gpt2_attn_heads_cma_top10": "2heads",
        "gpt2_attn_heads_dm_top10": "2heads",
        "gpt2_attn_heads_acdc": "2heads",
        "gpt2_attn_layers": "1layers",
        "gpt2_attn_layers_all": "1layers",
    }

    order = {
        "gpt2_baseline": "0",
        "gpt2_all": "1",
        "gpt2_random": "2",
        "gpt2_acdc": "5",
        "gpt2_attn_heads_cma_top10": "7",
        "gpt2_attn_heads_dm_top10": "8",
        "gpt2_attn_heads_acdc": "6",
        "gpt2_attn_layers": "4",
        "gpt2_attn_layers_all": "3",
    }


    results_path = Path("results/evaluation")
    list_of_files = list(results_path.glob('*.json'))
    results = []
    for f in list_of_files:
        results.append(pl.read_ndjson(f))
    results = pl.concat(results)
    results = results.filter(pl.col("model").is_in(filter).is_not())
    results = results.with_columns([
       pl.col("model").apply(lambda x: baselines[x]).alias("model_type"),
        pl.col("model").apply(lambda x: labels[x]).alias("model_label"),
        pl.col("model").apply(lambda x: order[x]).alias("model_order"),
    ])
     
    results = results.with_columns([
        ((pl.col("BLiMP ISV1") + pl.col("BLiMP RSV1") + pl.col("BLiMP ISV2") + pl.col("BLiMP RSV2"))/4).alias("BLiMP SV")])
   
    results = results.with_columns([
        ((pl.col("WinoBias Type1 Dev") + pl.col("WinoBias Type1 Test"))/2).alias("WinoBias1"),
        ((pl.col("WinoBias Type2 Dev") + pl.col("WinoBias Type2 Test"))/2).alias("WinoBias2"),
        ((pl.col("WinoBias Type1 Dev") + pl.col("WinoBias Type1 Test") + pl.col("WinoBias Type2 Dev") + pl.col("WinoBias Type2 Test"))/4).alias("WinoBias")
        ])
    return results

    
def get_perplexity_results():
    filter = [] #["gpt2_attn_layers","gpt2_attn_layers_all"]
    labels = {
        "gpt2_baseline": "gpt2",
        "gpt2_all": "full model",
        "gpt2_random": "random attn heads",
        "gpt2_acdc": "acdc",
        "gpt2_attn_heads_cma_top10": "cma attn heads",
        "gpt2_attn_heads_dm_top10": "dm attn heads",
        "gpt2_attn_heads_acdc": "acdc attn heads",
        "gpt2_attn_layers": "last 4 attn layers",
        "gpt2_attn_layers_all": "all attn layers",
    }
    baselines  = {
        "gpt2_baseline": "0baseline",
        "gpt2_all": "0baseline",
        "gpt2_random": "0baseline",
        "gpt2_acdc": "1layers",
        "gpt2_attn_heads_cma_top10": "2heads",
        "gpt2_attn_heads_dm_top10": "2heads",
        "gpt2_attn_heads_acdc": "2heads",
        "gpt2_attn_layers": "1layers",
        "gpt2_attn_layers_all": "1layers",
    }

    order = {
        "gpt2_baseline": "0",
        "gpt2_all": "1",
        "gpt2_random": "2",
        "gpt2_acdc": "5",
        "gpt2_attn_heads_cma_top10": "7",
        "gpt2_attn_heads_dm_top10": "8",
        "gpt2_attn_heads_acdc": "6",
        "gpt2_attn_layers": "4",
        "gpt2_attn_layers_all": "3",
    }


    results_path = Path("results/evaluation/perplexity")
    list_of_files = list(results_path.glob('wikitext_*.json'))
    results = []
    for f in list_of_files:
        results.append(pl.read_ndjson(f))
    results = pl.concat(results)
    results = results.filter(pl.col("model").is_in(filter).is_not())
    
     
    return results


def plot_attn_heads():
    cma = Path("results/cma") / "gpt2-small_attn_heads_10_seed_9.json"
    fig, cma_heads = plot_cma_heads(cma, 7)
    pio.write_image(fig, Path("results/plots")/f"heads_cma.png", format='png', width=1000, height=1000)

    diffmask = Path("results/discovery") / "gpt2-small_attn_heads_10_mlps_0_epochs_200_lr_0.001_seed_9.json"
    fig, dm_heads = plot_diffmask_heads(diffmask, 7)
    pio.write_image(fig, Path("results/plots")/f"heads_dm.png", format='png', width=1000, height=1000)

    acdc = Path("components/acdc.yaml")
    fig, acdc_heads = plot_acdc_heads(acdc, 10)
    pio.write_image(fig, Path("results/plots")/f"heads_acdc.png", format='png', width=1000, height=1000)

    fig = make_subplots(rows=1, cols=3, start_cell="bottom-left", 
                        subplot_titles=("CMA", "DiffMask", "ACDC"),
                        x_title='Heads',
                        y_title='Layers',
                        shared_yaxes=True)
    
    fig.add_trace(go.Heatmap(z=cma_heads, colorscale='Blues', showscale=False),
              row=1, col=1)

    fig.add_trace(go.Heatmap(z=dm_heads, colorscale='Blues', showscale=False),
              row=1, col=2)


    fig.add_trace(go.Heatmap(z=acdc_heads,  colorscale=['#8E44AD', '#2ECC71'], showscale=False),
              row=1, col=3)
    

    fig.update_layout(font=dict(size=20))
    fig.update_annotations(font=dict(size=20))

    fig.write_image(f"results/plots/heads.png", format='png', width=1800, height=600)

def plot_attn_heads_overlap():
    acdc, acdc_heads = get_heads("components/acdc.yaml")
    cma, cma_heads = get_heads("components/attn_heads_cma_top10.yaml")
    dm, dm_heads = get_heads("components/attn_heads_dm_top10.yaml")
    overlap = defaultdict(list)
    for l in acdc_heads.keys():
        for h in acdc_heads[l]:
            if l in cma_heads.keys() and h in cma_heads[l] and l in dm_heads.keys() and h in dm_heads[l]:
                overlap[l].append(h)
    overlap = dict(overlap)
    for m in [acdc, cma, dm]:
        for l in overlap.keys():
            for h in overlap[l]:
                m[int(l), int(h)] = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = ['#636EFA', '#FFFFFF', '#EF553B']
    my_cmap = ListedColormap(colors, name="my_cmap")

    # Create heatmaps in each subplot
    heatmap1 = axes[0].imshow(cma, cmap=my_cmap, aspect='equal', origin='lower')
    heatmap2 = axes[1].imshow(dm, cmap=my_cmap, aspect='equal',  origin='lower')
    heatmap3 = axes[2].imshow(acdc, cmap=my_cmap, aspect='equal',  origin='lower')

    # Set titles for subplots
    axes[0].set_title('CMA', fontsize=16)
    axes[1].set_title('DiffMask', fontsize=16)
    axes[2].set_title('ACDC', fontsize=16)

    axes[0].set_ylabel("Layers", fontsize=20)
    axes[1].set_xlabel('Heads', fontsize=20)

    font_properties = { 'size': 14}

    for ax in axes.flat:
        ax.tick_params(axis='both', which='both', labelsize=10)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontproperties(font_properties)

    # Adjust layout and display the plot
    plt.tight_layout(pad=2.0,)
    plt.savefig(f"results/plots/heads.pdf", format='pdf', dpi=300)

def get_heads(filepath):
    with open(filepath, 'r') as f:
        components = yaml.safe_load(f)
        heads = components['attn_heads']['subset']
        arr = np.zeros((12, 12))
        for l in heads.keys():
            for h in heads[l]:
                arr[int(l), int(h)] = -0.5
        arr = np.array(arr)
    return arr, heads


def print_latex(df, label, caption=f"Comparision of different models on the CrowS-Pairs, StereoSet, BLiMP and WinoBias datasets."):    
    cmap = 'RdYlGn'
    latex = df.style.background_gradient(cmap=cmap,
                                         axis=0).format(precision=3).to_latex(
        clines="skip-last;data",
        hrules=True, 
        convert_css=True,
        label=label,
        caption=caption,
          position_float="centering")
    print(latex)

    

if __name__ == '__main__':
    # plot_results()
    # plot_attn_heads_overlap()
    #plot_attn_heads()
    # perplexity_results = get_perplexity_results()
    # perplexity_results = perplexity_results.groupby(['model', 'model_type', 'model_label', 'model_order']).agg([pl.mean('PPL')]).sort(['model_order'])
    # perplexity_results = perplexity_results.select([pl.col("model"), pl.col("PPL").alias('WikiText PPL')])

    results = get_results()
    results = results.select(['model', 'seed', 'model_label', 'CrowS-Pairs', 'BLiMP', 'BLiMP AGA',  'BLiMP SV', 'WinoBias2', 'Professions'])
    results = results.to_pandas()
    # results = results.groupby(['model', 'model_type', 'model_label', 'model_order']).agg([pl.mean('CrowS-Pairs'),
    #                                                                                       pl.mean('BLiMP'),
    #                                                                                       pl.mean('BLiMP AGA'), 
    #                                                                                       pl.mean('BLiMP SV'),
    #                                                                                       pl.mean('WinoBias2'),
    #                                                                                       pl.mean('Professions')])
    # results = results.sort(['model_order'])
    # results = results.join(perplexity_results, on=['model'])
    # results = results.select(['model_label', 'WikiText PPL', 'BLiMP','BLiMP SV', 'BLiMP AGA', 'CrowS-Pairs', 'WinoBias2', 'Professions' ]).to_pandas()
    #results.set_index('model_label', inplace=True)
    results.to_csv("results/plots/results_all.csv")
    print(results.head())


    # # data = {'WinoBias': [0.9, 0.5, 0.2, 0.6],
    # #         'CrowS-Pairs': [0.99, 0.98, 0.95, 0.90]}
    
    # # df = pd.DataFrame(data, index=['GPT-2',
    # #                             'ACDC',
    # #                             'CMA',
    # #                             'DiffMask'])
    # #print_latex(results, "tab:smth")

    

