# %% [markdown]
# # Lab 06 — Neural networks from the inside
#
# Deep learning is built from one simple unit, the artificial neuron, repeated
# and stacked. This lab opens the box:
#
# 1. Build a single neuron by hand
# 2. Train a small neural network on field-trial data and watch it learn
# 3. See overfitting happen, and stop it
# 4. Compare with the random forest from Lab 03
#
# On tabular data like this, neural networks rarely beat tree models. They shine
# on images, sequences and text — which is where Lab 07 goes next.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

REPO = "YOUR-GITHUB-USERNAME/ai-for-agri-bio"


def data_path(name):
    """Use the local data folder if it exists, otherwise download from GitHub."""
    for folder in ["data", "../data", "../../data"]:
        if os.path.exists(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return f"https://raw.githubusercontent.com/{REPO}/main/data/{name}"


plt.rcParams["figure.figsize"] = (8, 4.5)
SEED = 42


def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# %% [markdown]
# ## 1. One neuron
#
# A neuron multiplies each input by a **weight**, adds them up with a **bias**,
# and passes the sum through an **activation function**. With a sigmoid
# activation, the output is a probability between 0 and 1 — this single neuron
# is exactly logistic regression.
#
# Here is a hand-made neuron that estimates disease risk from rainfall and
# temperature. The weights are made up; training would learn them from data.

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def neuron(rainfall_mm, temp_c, w_rain=0.006, w_temp=0.35, bias=-8.0):
    z = w_rain * rainfall_mm + w_temp * temp_c + bias
    return sigmoid(z)


for rain, temp in [(300, 12), (550, 15), (750, 18)]:
    print(f"rainfall {rain} mm, {temp} °C -> disease risk {neuron(rain, temp):.2f}")

# %%
z = np.linspace(-6, 6, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(z, sigmoid(z))
axes[0].set(title="Sigmoid: squashes to 0–1", xlabel="weighted sum z", ylabel="output")
axes[1].plot(z, np.maximum(0, z))
axes[1].set(title="ReLU: the default in deep networks", xlabel="weighted sum z", ylabel="output")
plt.tight_layout()
plt.show()

# %% [markdown]
# A **network** connects many neurons in layers. Each hidden neuron learns its
# own combination of the inputs; the output neuron combines the hidden ones.
# Training adjusts all the weights, a little at a time, to reduce the
# prediction error (gradient descent with backpropagation). One pass through
# the training data is an **epoch**.
#
# ## 2. Train a network on field-trial data

# %%
trials = pd.read_csv(data_path("crop_trials.csv"))
trials.loc[trials["yield_t_ha"] > 20, "yield_t_ha"] = np.nan
trials.loc[trials["soil_ph"] < 3, "soil_ph"] = np.nan
trials = trials.dropna(subset=["yield_t_ha"])

features = ["n_fert_kg_ha", "soil_p_mg_kg", "soil_k_mg_kg", "soil_ph", "organic_matter_pct",
            "rainfall_mm", "mean_temp_c", "irrigated", "variety"]
X = pd.get_dummies(trials[features], columns=["variety"], dtype=int)
X = X.fillna(X.median())
y = trials["yield_t_ha"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

# Neural networks need inputs on similar scales: standardise to mean 0, SD 1.
# The scaler learns its means and SDs from the training data only.
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# %% [markdown]
# We train one epoch at a time and record the error on training and test data.


# %%
def training_curve(hidden, alpha, epochs=300):
    net = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=0.01,
                       batch_size=32, random_state=SEED)
    train_curve, test_curve = [], []
    for _ in range(epochs):
        net.partial_fit(X_train_s, y_train)
        train_curve.append(rmse(y_train, net.predict(X_train_s)))
        test_curve.append(rmse(y_test, net.predict(X_test_s)))
    return net, train_curve, test_curve


net, tr, te = training_curve(hidden=(16,), alpha=0.0001)
plt.plot(tr, label="training data")
plt.plot(te, label="test data")
plt.ylim(0, 3)
plt.xlabel("epoch")
plt.ylabel("RMSE (t/ha)")
plt.title("A network with 16 hidden neurons learning")
plt.legend()
plt.show()

# %% [markdown]
# The error drops fast in the first epochs as the network finds the main
# patterns, then improves slowly.
#
# ## 3. Overfitting, and how to stop it
#
# A big network with no brakes can memorise the training data. Two common
# brakes are **weight decay** (L2 regularisation, `alpha`: penalise large
# weights) and **early stopping** (stop when validation error stops improving).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (alpha, title) in zip(axes, [(0.0, "256 neurons, no regularisation"),
                                     (5.0, "256 neurons, strong weight decay")]):
    _, tr, te = training_curve(hidden=(256,), alpha=alpha)
    ax.plot(tr, label="training data")
    ax.plot(te, label="test data")
    ax.set(title=title, xlabel="epoch", ylim=(0, 3))
    ax.legend()
axes[0].set_ylabel("RMSE (t/ha)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Without regularisation, training error keeps falling while test error
# creeps up: the network is memorising noise. With weight decay, the gap
# between training and test error stays small.
#
# `MLPRegressor(early_stopping=True)` does early stopping automatically, using
# 10% of the training data as a validation set.
#
# ## 4. Compare with a random forest

# %%
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, early_stopping=True,
                   max_iter=2000, random_state=SEED).fit(X_train_s, y_train)
forest = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X_train, y_train)

pd.DataFrame({
    "model": ["baseline (mean)", "neural network (64-32)", "random forest"],
    "test RMSE (t/ha)": [rmse(y_test, np.full(len(y_test), y_train.mean())),
                         rmse(y_test, mlp.predict(X_test_s)),
                         rmse(y_test, forest.predict(X_test))],
}).round(3)

# %% [markdown]
# The network does well, but the random forest is as good or better — with no
# scaling, no learning rate and no epochs to tune. **For tables of measurements,
# start with tree models.** Use deep learning when the input is an image, a
# sequence or text, where it can learn the features itself.
#
# **Try it:** Change `hidden_layer_sizes` to `(4,)` and `(256, 128, 64)`. When
# does the network underfit, and when does it overfit?
