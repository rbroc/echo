import pathlib
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def explore_data(model="beluga7b", file="stories_prompt_1.ndjson", vllm=False, root_data_path=pathlib.Path(__file__).parents[2] / "datasets"):
    if vllm:
        datapath = root_data_path / "ai_vllm_datasets" / "ALL_DATA"
    else: 
        datapath = root_data_path / "ai_datasets" / "ALL_DATA"

    # load data 
    filepath = datapath / model / file 
    df = pd.read_json(filepath, lines=True)

    # get mean length 
    mean_length =  df[f"{model}_completions"].apply(len).mean()
    print(f"Mean Length: {mean_length}")

    return df 

def plot_string_length_distribution_from_df(df, model="beluga7b", col="completions", framework_col='framework', bins=30, figsize=(10, 6), title='String Lengths by Framework', save_path=None):
    full_column_name = f"{model}_{col}"
    if full_column_name not in df.columns or framework_col not in df.columns:
        raise ValueError(f"Required columns not found in DataFrame.")

    df['string_length'] = df[full_column_name].astype(str).apply(len)

    # Set the style for a prettier plot
    sns.set(style="whitegrid")

    # Plotting the histogram
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x='string_length', hue=framework_col, kde=False, bins=bins, palette='viridis')
    plt.title(title)
    plt.xlabel('Length of Strings')
    plt.ylabel('Frequency')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def main(): 
    path = pathlib.Path(__file__)

    model = "beluga7b"

    # plot 
    vllm = explore_data(vllm=True)
    hf = explore_data(vllm=False)

    # add framework 
    vllm["framework"] = "vllm"
    hf["framework"] = "hf"

    # drop cols for both
    for df in [vllm, hf]:
        df.drop(columns=["prompt_1"], inplace=True)

    # concatenate 
    df = pd.concat([vllm, hf], axis=0)

    # plot 
    plot_string_length_distribution_from_df(df, save_path=path.parents[2] / "results" / "string_length_distribution.png")

    print(df)

if __name__ == "__main__":
    main()
    