import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

excel_file = "Housing.csv"
price_column = "price"
variable_columns = ["area","bedrooms","bathrooms","stories",
                    "mainroad", "guestroom", "basement", "parking"]

learnRate = 0.001
epoch = 200
size = 64



def relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return (x > 0).astype(float)

script_direction = os.path.dirname(os.path.abspath(__file__))
print(f"\nLooking for file in: {script_direction}")

excel_path = os.path.join(script_direction, excel_file)

print(f"Attepmting to load dataset: {excel_file} please wait\n")

if not os.path.exists(excel_path):
    raise FileNotFoundError(
        f"\nCould not find {excel_file}"
        f"\nMake sure {excel_file} is in the same folder as this script"
    )
    
df = pd.read_excel(excel_path)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

missing = [z for z in variable_columns + [price_column] if z not in df.columns]
if missing:
    raise ValueError(
        f"\nThese columns weren't found in the dataset: {missing}\n"
    )

df[variable_columns] = df[variable_columns].fillna(df[variable_columns].mean())
df[price_column] = df[price_column].fillna(df[price_column].mea())

X = df[variable_columns].values
y = df[price_column].values.reshape(-1,1)
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, randomState = 42
)