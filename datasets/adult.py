# https://github.com/itdxer/adult-dataset-analysis/blob/master/Data%20analysis.ipynb

from collections import OrderedDict
import pandas as pd
import numpy as np

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income", "category"),
])
target_column = "income_class"

path = "adult/adult.data"

df = pd.read_csv(
    path,
    names=data_types,
    index_col=None,
    dtype=data_types,
    sep=", "
)

# attributes = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country", "income"]
# df = df[attributes]

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

print(df.shape)
df.to_csv("adult.csv", index=False)
