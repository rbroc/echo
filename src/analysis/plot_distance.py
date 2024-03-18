import pathlib
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def split_into_sentences(text):
    '''
    helper function for creating interactive jitterplots where the text is wrapped nicely.
    '''
    sentences = re.split(r'(?<=\w\.\s|\w\.\n)', text)  # split on dots followed by space or newline
    return "<br>".join(sentences)

def interactive_jitterplot(df:pd.DataFrame, save_path=None, save_file_name="distances.html"):
    '''
    Create interactive plots using plotly where distances are grouped by prompt and model for each dataset 

    Args
        data: data with euclidean distances
        datasets: list of datasets for which you want a plot
        save_path: if unspecified, no plot is saved. Will ensure that the specified path is created if it does not exist already.
    '''
    # split text into full sentences with line breaks for hover
    df["completions"] = df["completions"].astype(str)
    df["completions"] = df["completions"].apply(split_into_sentences)

    # round distance col
    df['distance'] = df['distance'].round(3)

    # create a new column to represent model with different colors
    df['dataset'] = df['dataset'].astype(str)

    fig = px.strip(df, x="model", y="distance", color="dataset", title=f"Distances to human across models", 
                       custom_data=["model", "id", "distance", "completions", "temperature"])

    fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
    fig.update_xaxes(categoryorder='total ascending')
    fig.update_layout(legend_title_text='Dataset')

    # appearance (custom hover template to display the completions text, ID, and model name)
    hovertemplate = "MODEL: %{customdata[0]}<br>ID: %{customdata[1]}<br>DISTANCE: %{customdata[2]}<br>%{customdata[3]}<br>TEMPERATURE: %{customdata[4]}<extra></extra>"
    fig.update_traces(hovertemplate=hovertemplate)

    # fix title
    fig.update_layout(title_x=0.5, title_font=dict(size=24))

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / save_file_name
        
        with open(save_file, "w") as file:
            file.write(fig.to_html(full_html=False, include_plotlyjs=True))
        file.close()

def unique_no_nan(x):
    return x.dropna().unique()

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "analysis"

    df = pd.read_csv(datapath / "distances_PC_cols.csv", index_col=False)

    # create plot per temp (but exclude nan)
    for temp in sorted(df["temperature"].dropna().unique()):
        # filter df 
        temp_df = df[(df["temperature"] == temp) | (df["model"] == "human")]
        interactive_jitterplot(temp_df, save_path=datapath, save_file_name=f"distances_temp{temp}.html")


if __name__ == "__main__":
    main()