"""
Formatter for loading and processing splits of data
"""
import pathlib
import pickle
from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SplitFormatter:
    def __init__(
        self,
        splits_dir: pathlib.Path,
        dataset: str = "dailymail_cnn",
        splits_to_load: List[str] = ["train", "val", "test"],
    ):
        self.dataset = dataset
        self.splits_dir = splits_dir
        self.splits_to_load = splits_to_load
        self.splits = {}

    def loader_function(self, split: str):
        print(f"[INFO]: Dummy loader function called for split '{split}'.")

    def filter_split(self, split, column:str, value:str):
        split_data = self.get_split(split)

        if split_data.empty:
            print(f"[WARNING]: Split '{split}' is empty.")
            return pd.DataFrame()

        self.splits[split] = split_data[split_data[column] == value] # consider making it so that you can filter on multiple values

        return self.splits[split]

    def load_splits(self):
        for split in self.splits_to_load:
            self.splits[split] = self.loader_function(split)

            if self.splits[split] is not None:
                print(f"[INFO]: Loaded data for split '{split}'. Filtering on {self.dataset}.")
                self.splits[split] = self.splits[split][self.splits[split]["dataset"] == self.dataset]
            else:
                print(f"[WARNING]: No data loaded for split '{split}'.")

        return self.splits  

    def get_split(self, split_name: str):
        return self.splits.get(split_name, pd.DataFrame())

    def get_X_y_data(self, split_name: str, X_col: str, y_col: str):
        """
        Get X and y data for a given split
        """
        split_data = self.get_split(split_name)
        if split_data.empty:
            print(f"[WARNING]: Split '{split_name}' is empty.")
            return np.array([]), np.array([])
        
        X = split_data[X_col].values
        y = split_data[y_col].values

        return X, y

    def get_X_y_sample(self, split_name:str, X_col:str, y_col:str, sample_size:int=5, random_state:int=129, stratify:bool=True):
        """
        Get a sample of X and y data for a given split
        """
        split_data = self.get_split(split_name)

        if split_data.empty:
            print(f"[WARNING]: Split '{split_name}' is empty.")
            return np.array([]), np.array([])
        
        # split up into the two classes
        class_0 = split_data[split_data[y_col] == 0]
        class_1 = split_data[split_data[y_col] == 1]

        # sample from each class
        sample_size_class_0 = sample_size // 2
        sample_size_class_1 = sample_size - sample_size_class_0

        sample_class_0 = class_0.sample(n=sample_size_class_0, random_state=random_state)
        sample_class_1 = class_1.sample(n=sample_size_class_1, random_state=random_state)

        sample_df = pd.concat([sample_class_0, sample_class_1]).sample(frac=1, random_state=random_state)

        X_sample = sample_df[X_col].values
        y_sample = sample_df[y_col].values  

        return X_sample, y_sample

class TextSplitFormatter(SplitFormatter):
    def __init__(
        self,
        splits_dir: pathlib.Path,
        dataset = "dailymail_cnn",
        splits_to_load: List[str] = ["train", "val", "test"],
    ):
        super().__init__(splits_dir, dataset, splits_to_load)

    def loader_function(self, split: str):
        file_path = self.splits_dir / f"{split}_text.parquet"
        try:
            data = pd.read_parquet(file_path)
            return data
        except FileNotFoundError:
            print(f"[ERROR]: File '{file_path}' not found.")
            return pd.DataFrame()

    def vectorise(
        self,
        X_col: str = "completions",
        split_to_fit_on: str = "train",
        splits_to_transform: List[str] = ["val", "test"],
        max_features: int = 1000,
    ):
        main_split = self.get_split(split_to_fit_on)
        
        if not main_split.empty:
            vectorizer = TfidfVectorizer(lowercase=False, max_features=max_features)
            main_split["tfidf"] = list(vectorizer.fit_transform(main_split[X_col]).toarray())
            self.splits[split_to_fit_on] = main_split

            for split_name in splits_to_transform:
                split_data = self.get_split(split_name)
                if not split_data.empty:
                    split_data["tfidf"] = list(vectorizer.transform(split_data[X_col]).toarray())
                    self.splits[split_name] = split_data

class EmbeddingSplitFormatter(SplitFormatter):
    def __init__(
        self,
        splits_dir: pathlib.Path,
        dataset = "dailymail_cnn",
        splits_to_load: List[str] = ["train", "val", "test"],
    ):
        super().__init__(splits_dir, dataset, splits_to_load)

    def loader_function(self, split: str):
        embeddings_path = self.splits_dir / f"{split}_embeddings.npy"
        meta_data_path = self.splits_dir / f"{split}_metadata.parquet"

        try:
            embeddings = np.load(embeddings_path)
            meta_data = pd.read_parquet(meta_data_path)

            # add embeddings to metadata (to filter by dataset later)
            meta_data["embeddings"] = embeddings.tolist()
            data = meta_data.copy()

            return data

        except FileNotFoundError:
            print(f"[ERROR]: File '{embeddings_path}' not found.")
            return pd.DataFrame()

class MetricsSplitFormatter(SplitFormatter): 
    def __init__(
        self,
        splits_dir: pathlib.Path,
        dataset = "dailymail_cnn",
        splits_to_load: List[str] = ["train", "val", "test"],
    ):
        super().__init__(splits_dir, dataset, splits_to_load)

    def loader_function(self, split: str):
        file_path = self.splits_dir / f"{split}_metrics.parquet"
        try:
            data = pd.read_parquet(file_path)
            return data
        except FileNotFoundError:
            print(f"[ERROR]: File '{file_path}' not found.")
            return pd.DataFrame()
    
    def get_X_and_y_data(self, split_name: str, X_features: List[str], y_col: str, pca_model_path: str, scaler_path: str):
        split_data = self.get_split(split_name)
        if split_data.empty:
            print(f"[WARNING]: Split '{split_name}' is empty.")
            return np.array([]), np.array([])

        # load PCA model and scaler
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)

        with open(pca_model_path, "rb") as file:
            pca_model = pickle.load(file)

        print(f"[INFO]: Transforming X with Scaler and PCA for split '{split_name}'.")
        X_scaled = scaler.transform(split_data[X_features])
        X_transformed = pca_model.transform(X_scaled)
        y = split_data[y_col].values

        return X_transformed, y