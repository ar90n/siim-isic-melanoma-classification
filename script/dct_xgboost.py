# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from siim_isic_melanoma_classification import io
from siim_isic_melanoma_classification.dataset import MelanomaDataset
from siim_isic_melanoma_classification.datasource import get_folds_by
from siim_isic_melanoma_classification.transforms import ZigZagFlatten

# %%
from torchtoolbox.transform import transforms
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# %%
fold_index = 0
n_fold = 4

# %%
all_source, _ = io.load_my_isic2020_csv(size=128, is_sanity_check=False)
train_source, val_source = get_folds_by(all_source, fold_index, n_fold)
train_source.df = train_source.df.iloc[:4096]
val_source.df = val_source.df.iloc[:4096]

# %%
train_dataset = MelanomaDataset(
    source=train_source,
    train=True,
    meta_features=["sex", "age_approx", "anatom_site_general_challenge"],
    dct=True,
    transforms=transforms.Compose({ZigZagFlatten(8)}),
)

val_dataset = MelanomaDataset(
    source=val_source,
    train=True,
    meta_features=["sex", "age_approx", "anatom_site_general_challenge"],
    dct=True,
    transforms=transforms.Compose({ZigZagFlatten(8)}),
)


# %%
x_train, y_train = zip(*[(np.hstack(v[0]), v[1]) for v in train_dataset])
x_train = np.stack(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)

x_val, y_val = zip(*[(np.hstack(v[0]), v[1]) for v in val_dataset])
x_val = np.stack(x_val)
y_val = np.array(y_val)
print(x_val.shape, y_val.shape)

# %%
clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=8192)

clf.fit(
    x_train,
    y_train,
    eval_metric="auc",
    eval_set=[(x_train, y_train), (x_val, y_val),],
    early_stopping_rounds=64,
    callbacks=[wandb_callback()],
)

y_hat = clf.predict_proba(x_val)[:, 1]

auc = roc_auc_score(y_val, y_hat)
print(f"auc: {auc}")
# %%
