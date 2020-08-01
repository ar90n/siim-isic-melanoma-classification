# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

from siim_isic_melanoma_classification import util, io, dataset, datasource

size = 256

# +
train_2020_df, test_2020_df = io.load_jpeg_melanoma_dataframe(size, True)
train_2019_df = io.load_jpeg_isic2019_dataframe(size, True)


train_2020_df["dataset"] = "isic2020"
test_2020_df["dataset"] = "isic2020"
train_2019_df["dataset"] = "isic2019"

all_df = pd.concat(
    [train_2020_df, train_2019__df, test_2020_df], 
)
all_df = all_df.drop(['diagnosis', 'benign_malignant', 'tfrecord', 'width', "height" ], axis=1)
dups = [
"ISIC_0026989",
"ISIC_0014360_downsampled",
"ISIC_0014289_downsampled",
"ISIC_0028003",
"ISIC_0032245",
"ISIC_0014513_downsampled",
"ISIC_0014507_downsampled",
"ISIC_0014331_downsampled",
"ISIC_0028760",
"ISIC_0025316",
"ISIC_0014478_downsampled",
"ISIC_0030366",
"ISIC_0014369_downsampled",
"ISIC_0027547",
"ISIC_0026309",
"ISIC_0027759",
"ISIC_0024822",
"ISIC_0014386_downsampled",
"ISIC_0028697",
"ISIC_0012547_downsampled",
"ISIC_0032469",
"ISIC_0012551_downsampled",
"ISIC_0031383",
"ISIC_0030667",
"ISIC_0014585_downsampled",
"ISIC_0016053_downsampled",
"ISIC_0012523_downsampled",
"ISIC_0014311_downsampled",
"ISIC_0025059",
"ISIC_0031516",
"ISIC_0016071_downsampled",
"ISIC_0024878",
"ISIC_0031320",
"ISIC_0028729",
"ISIC_0027484",
"ISIC_0029076",
"ISIC_0031072",
"ISIC_0014433_downsampled",
"ISIC_0027660",
"ISIC_0012526_downsampled",
"ISIC_0032314",
"ISIC_0014299_downsampled",
"ISIC_0031326",
"ISIC_0030017",
"ISIC_0034072",
"ISIC_0031107",
"ISIC_0028008",
"ISIC_0028469",
"ISIC_0024337",
"ISIC_0026016",
"ISIC_0014516_downsampled",
"ISIC_0014409_downsampled",
"ISIC_0029940",
"ISIC_0028262",
"ISIC_0026955",
"ISIC_0025202",
"ISIC_0059267",
"ISIC_0025570",
"ISIC_0028283"
]
all_df.drop(all_df[all_df["image_name"].isin(dups)].index)


replace_tbl = {
    "anatom_site_general_challenge": {
            'posterior torso': 'torso',
    'lateral torso': 'torso',
    'anterior torso': 'torso'
    },
    "sex": {
        'male': 1,
        'female': -1, 
        'unknown': 0,
        'nan': 0
    },
}
all_df = all_df.replace(replace_tbl)

one_hot_anatomies = pd.get_dummies(all_df["anatom_site_general_challenge"], dummy_na=True, dtype=np.uint8, prefix="site")
all_df = pd.concat([all_df, one_hot_anatomies], axis=1)

all_df["age_approx"] /= 100.0
all_df["age_approx"] = all_df["age_approx"].fillna(0)
all_df["patient_id"] = all_df["patient_id"].fillna(0)

train_rows = (train_2020_df.shape[0] + train_2019__df.shape[0])
train_df = all_df.iloc[: train_rows]
test_df = all_df.iloc[train_rows:]
test_df = test_df.drop("target", axis=1)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(8, shuffle=True)
train_df["fold"] = -1
for i, (_, idx) in enumerate(kfold.split(train_df, train_df["target"])):
    train_df["fold"].iloc[idx] = i

    
train_df["sanity_check"] = 0
test_df["sanity_check"] = 0
train_df.iloc[train_df.query("target == 1").index[:16], train_df.columns.get_loc('sanity_check')] = 1
test_df.iloc[:16]['sanity_check'] = 1
# -

train_df

test_df



train_source, test_source = io.load_my_isic2020_csv()
train_dataset = dataset.MelanomaDataset(train_source, train = True)

train_source, val_source = datasource.train_validate_split(train_source, is_sanity_check=True)

train_source.df["fold"].unique()

val_source.df

for train_source, val_source in datasource.kfold_split(train_df, 8, True):
    print(train_source.folds, val_source.folds)

train_source.df[train_source.df[train_source.df["sanity_check"] == 1]["fold"].isin([1,2])]


