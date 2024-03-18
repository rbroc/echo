'''
Plot eucludean distances (both a static and dynamic version)
'''
import pathlib
import pandas as pd

import sys 
sys.path.append(str(pathlib.Path(__file__).parents[3]))
from src.utils.distance import jitterplots, interactive_jitterplot

def main(): 
    path = pathlib.Path(__file__)
    datapath = path.parents[3] / "results" / "prompt_select" / "distance"

    df = pd.read_csv(datapath / "distances_all_PC_cols.csv")
    
    models = ["beluga", "llama2_chat"]
    datasets = ["dailymail_cnn", "stories", "mrpc", "dailydialog"]

    # create normal plot 
    jitterplots(df, datasets, datapath / "all_PC_jitterplots" / "static") 

    # create interactive plots 
    interactive_jitterplot(df, datasets, datapath / "all_PC_jitterplots" / "interactive") 

if __name__ == "__main__":
    main()