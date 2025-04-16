# data_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class AgriDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

        # Assume columns: ['feature1', ..., 'featureN', 'task_id', 'crop_label', 'fertilizer_label']
        self.feature_cols = [col for col in self.df.columns if col.startswith('feature')]

        self.label_enc_crop = LabelEncoder()
        self.label_enc_fert = LabelEncoder()

        # Encode target labels
        self.df['crop_label'] = self.label_enc_crop.fit_transform(self.df['crop_label'])
        self.df['fertilizer_label'] = self.label_enc_fert.fit_transform(self.df['fertilizer_label'])

        # Normalize features
        self.scaler = StandardScaler()
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])

        self.input_dim = len(self.feature_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        task_id = torch.tensor(int(row['task_id']), dtype=torch.long)
        crop_target = torch.tensor(int(row['crop_label']), dtype=torch.long)
        fert_target = torch.tensor(int(row['fertilizer_label']), dtype=torch.long)
        return features, task_id, crop_target, fert_target
