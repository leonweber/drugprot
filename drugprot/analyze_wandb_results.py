import numpy as np
import pandas as pd

important_settings = [
    "config_model/aggregate_after_logits",
    "config_model/blind_entities",
    "config_model/entity_embeddings",
    "config_model/entity_side_information",
    "config_model/loss",
    "config_model/mark_with_special_tokens",
    "config_model/pair_side_information",
    "config_model/transformer",
    "config_model/tune_thresholds",
    "config_model/use_cls",
    "config_model/use_doc_context",
    "config_model/use_ends",
    "config_model/use_starts",
    "config_data/train"
]

base_hyperparams = [
    "config_model/lr",
    "config_data/dataset_to_batch_size/drugprot",
    "config_model/max_length",
    "config_trainer/max_epochs",
    "config_seed"
]

if __name__ == '__main__':
    df = pd.read_csv("results.csv")
    df = df[~(df['end_drugprot/val/f1'].isna())]
    df = df[df['end_drugprot/val/f1'] < 0.81]
    for setting in important_settings:
        values = df[setting].unique()
        for value in values:
            try:
                if np.isnan(value):
                    continue
            except TypeError:
                pass

            df_setting = df[df[setting] == value]
            idx_max = df_setting['end_drugprot/val/f1'].argmax()
            print(f"{setting}: {value}")
            print(df_setting.iloc[idx_max]["config_out_dir"])
            print("===============================")
            print(df_setting.iloc[idx_max][base_hyperparams])
            print()