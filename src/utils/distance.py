'''
Compute euclidean distances between a baseline (typically human completions) and otehr models. Includes functions to plot them.
'''
import pathlib
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def compute_distances(df:pd.DataFrame, models:list=["beluga7b", "llama2_chat13b"], cols:list=["PC1", "PC2", "PC3", "PC4"], baseline="human"):
    '''
    Extract euclidean distances between human and model completions in n-dimensions from a list of features (cols) 

    Args
        df: dataframe with features columns (e.g., PC components)
        models: list of models present in the model column in the dataframe
        cols: list of feature cols present in the dataframe (e.g., PC components)
        baseline: model which the euclidean distance is computed between for all other models. Defaults to human completions.

    Returns
        result_df: dataframe containing columns: id, model, dataset, distance, prompt_number
    '''
    result_rows = []    

    # subset df to include only the AI models
    df_ai = df[df["model"].isin(models)]

    for _, row in df_ai.iterrows():
            # extract "id" for the current row
            current_id = row["id"]

            # extract features for the "human" model with the same "id" as df_ai 
            pc_human = df[(df["model"] == baseline) & (df["id"] == current_id)][cols].values

            # extract features for model completions
            pc_model = row[cols].values

            # compute euclidean distance in n-dimensions
            distance = np.sqrt(np.sum((pc_human - pc_model) ** 2))

            result_row = {
                "id": row["id"],
                "model": row["model"],
                "dataset": row["dataset"],
                "distance": distance,
                "prompt_number": row["prompt_number"],
                "completions": row["completions"]
            }

            result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows)

    return result_df

# plotting # 
def jitterplots(data:pd.DataFrame, datasets:list, save_path:pathlib.Path):
    for dataset in datasets:
        dataset_data = data[data['dataset'] == dataset]

        dataset_data['model'] = dataset_data['model'].astype(str)

        # create plot
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
    '''
    helper function for creating interactive jitterplots where the text is wrapped nicely.
    '''
    sentences = re.split(r'(?<=\w\.\s|\w\.\n)', text)  # split on dots followed by space or newline
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

        # round distance col
        dataset_data['distance'] = dataset_data['distance'].round(3)

        # create a new column to represent model with different colors
        dataset_data['model'] = dataset_data['model'].astype(str)

        fig = px.strip(dataset_data, x="prompt_number", y="distance", color="model", title=f"{dataset.upper()}", 
                       custom_data=["model", "id", "distance", "completions"])

        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.update_xaxes(categoryorder='total ascending')
        fig.update_layout(legend_title_text='Model')

        # appearance (custom hover template to display the completions text, ID, and model name)
        hovertemplate = "MODEL: %{customdata[0]}<br>ID: %{customdata[1]}<br>DISTANCE: %{customdata[2]}<br>%{customdata[3]}"
        fig.update_traces(hovertemplate=hovertemplate)

        # fix title
        fig.update_layout(title_x=0.5, title_font=dict(size=24))

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / f"{dataset}.html"
            fig.write_html(str(save_file))