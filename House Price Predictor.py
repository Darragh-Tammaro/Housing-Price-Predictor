import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

excel_file = "Housing.csv"
price_column = "price"
variable_columns = ["area","bedrooms","bathrooms","stories",
                    "mainroad", "guestroom", "basement", "parking"]

learnRate = 0.001
epochs = 400
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
    
df = pd.read_csv(excel_path)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
yes_or_no = ["mainroad","guestroom","basement"]
df[yes_or_no] = df[yes_or_no].apply(lambda col:col.map({"yes":1,"no":0}))

missing = [z for z in variable_columns + [price_column] if z not in df.columns]
if missing:
    raise ValueError(
        f"\nThese columns weren't found in the dataset: {missing}\n"
    )

df[variable_columns] = df[variable_columns].fillna(df[variable_columns].mean())
df[price_column] = df[price_column].fillna(df[price_column].mean())

X = df[variable_columns].values
y = df[price_column].values.reshape(-1,1)
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

print(f"\nSamples for training: {X_train.shape[0]:,}")
print(f"Samples for testing: {X_test.shape[0]:,}")
print(f"Feature for each sample {n_features}\n")

np.random.seed(42)
W1 = np.random.randn(n_features, 64) * np.sqrt(2 / n_features)
W2 = np.random.randn(64,32) * np.sqrt(2 / 64)
W3 = np.random.randn(32,1) * np.sqrt(2 / 32)

B1 = np.zeros((1,64))
B2 = np.zeros((1,32))
B3 = np.zeros((1,1))

losses = []
print("Training please wait\n")
num_samples = X_train.shape[0]

for epoch in range(epochs): 
    indices = np.random.permutation(num_samples)
    ShuffleX = X_train[indices]
    ShuffleY = y_train[indices]
    for start in range(0, num_samples, size):
        BatchX = ShuffleX[start:start + size]
        BatchY = ShuffleY[start:start + size]

        #forward pass starts here 

        Z1 = BatchX @ W1 + B1     #hidden 1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + B2     #hidden 2
        A2 = relu(Z2)

        Z3 = A2 @ W3 + B3    #output lay
        predictionY = Z3 

        #backwards pass starts

        d_Z3 = (predictionY - BatchY) / len(BatchY)
        dW3 = A2.T @ d_Z3
        dB3 = np.sum(d_Z3, axis=0, keepdims=True)

        d_A2 = d_Z3 @ W3.T
        d_Z2 = d_A2 * relu_deriv(Z2)
        dW2 = A1.T @ d_Z2
        dB2 = np.sum(d_Z2, axis=0, keepdims=True)

        d_A1 = d_Z2 @ W2.T
        d_Z1 = d_A1 * relu_deriv(Z1)
        dW1 = BatchX.T @ d_Z1
        dB1 = np.sum(d_Z1, axis=0, keepdims=True)

        W3 -= learnRate* dW3
        B3 -= learnRate * dB3
        W2 -= learnRate * dW2
        B2 -= learnRate * dB2
        W1 -= learnRate * dW1
        B1 -= learnRate * dB1
    
    Z1_f = X_train @ W1 + B1; A1_f = relu(Z1_f)
    Z2_f = A1_f @ W2 + B2; A2_f = relu(Z2_f)
    Z3_f = A2_f @ W3 + B3
    epochLoss = np.mean((y_train - Z3_f) ** 2)
    losses.append(epochLoss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch:>3} / {epochs} | Loss: {epochLoss:.4f}")

Z1_t = X_test @ W1 + B1; A1_t = relu(Z1_t)
Z2_t = A1_t @ W2 + B2; A2_t = relu(Z2_t)
Z3_t = A2_t @ W3 + B3

Yprediction_real = scaler_y.inverse_transform(Z3_t)
Ytest_real = scaler_y.inverse_transform(y_test)
mean_Unc = np.mean(np.abs(Ytest_real - Yprediction_real))


res_ss = np.sum((Ytest_real - Yprediction_real) ** 2)
tot_ss = np.sum((Ytest_real - np.mean(Ytest_real)) ** 2)
r2 = 1 - (res_ss / tot_ss)

print(f"\n --- Results ---")
print(f"Mean absolute uncertainty : ${mean_Unc:,.0f}")
print(f"R2 Score : {r2:.4f}   (where 1.0 is considered perfect)")
print(f" -------------")

        






