'''
Plot eucludean distances 
'''

import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_jitterplots(data, datasets, save_path):
    for dataset in datasets:
        dataset_data = data[data['dataset'] == dataset]

        # create lpot
        g = sns.catplot(data=dataset_data, x="prompt_number", y="distance", hue="model", kind="strip", palette="husl", jitter=0.3)

        # Set labels and title
        plt.subplots_adjust(top=0.9)
        g.set_axis_labels("Prompt Number", "Distance")
        g.fig.suptitle(f"{dataset.upper()}")

        # Save the plot
        if save_path: 
            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / f"{dataset}.png"
            g.savefig(save_file, dpi=600)

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[2] / "results" / "distance"

    df = pd.read_csv(datapath / "distances_all_PC_cols.csv")
    
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    create_jitterplots(df, datasets, datapath / "all_PC_jitterplots") 


if __name__ == "__main__":
    main()