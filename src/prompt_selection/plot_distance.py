'''
Plot eucludean distances (both a static and dynamic version)
'''
import pathlib

import pandas as pd
import re

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def jitterplots(data, datasets, save_path):
    for dataset in datasets:
        dataset_data = data[data['dataset'] == dataset]

        # create lpot
        g = sns.catplot(data=dataset_data, x="prompt_number", y="distance", hue="model", kind="strip", palette="husl", jitter=0.3)

        # set labels and title
        plt.subplots_adjust(top=0.9)
        g.set_axis_labels("Prompt Number", "Distance")
        g.fig.suptitle(f"{dataset.upper()}")

        if save_path: 
            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / f"{dataset}.png"
            g.savefig(save_file, dpi=600)

def split_into_sentences(text):
    sentences = re.split(r'(?<=\w\.\s|\w\.\n)', text)  # Split on dots followed by space or newline
    return "<br>".join(sentences)

def interactive_jitterplot(data:pd.DataFrame, datasets:list=["dailymail_cnn", "stories", "mrpc", "dailydialog"], save_path=None):
    '''
    Create interactive plots using plotly where distances are grouped by prompt and model for each dataset 

    Args
        data: data with euclidean distances
        datasets: list of datasets for which you want a plot
        save_path: if unspecified, no plot is saved. Will ensure that the specified path is created if it does not exist already.
    '''
    for dataset in datasets:
        dataset_data = data[data["dataset"] == dataset]

        # split text into full sentences with line breaks for hover
        dataset_data["completions"] = dataset_data["completions"].apply(split_into_sentences)

        # create a new column to represent model with different colors
        dataset_data['model'] = dataset_data['model'].astype(str)

        fig = px.strip(dataset_data, x="prompt_number", y="distance", color="model", title=f"{dataset.upper()}", 
                       custom_data=["completions", "id", "model"])

        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.update_xaxes(categoryorder='total ascending')
        fig.update_layout(legend_title_text='Model')

        # appearance
        # custom hover template to display the completions text, ID, and model name
        hovertemplate = "MODEL: %{customdata[2]}<br>ID: %{customdata[1]}<br>%{customdata[0]}"
        fig.update_traces(hovertemplate=hovertemplate)

        # fix title
        fig.update_layout(title_x=0.5, title_font=dict(size=24))

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / f"{dataset}.html"
            fig.write_html(str(save_file))

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "distance"

    df = pd.read_csv(datapath / "distances_all_PC_cols.csv")
    
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    # create normal plot 
    jitterplots(df, datasets, datapath / "all_PC_jitterplots" / "static") 

    # create interactive plots 
    interactive_jitterplot(df, datasets, datapath / "all_PC_jitterplots" / "interactive") 


if __name__ == "__main__":
    main()