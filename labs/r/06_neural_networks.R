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
#
# We use `nnet`, which ships with R and fits networks with one hidden layer.
# For deep networks in R, see `torch` or `keras3` (Lab 07A).

# %%
pkgs <- c("ggplot2", "nnet", "randomForest")
new <- setdiff(pkgs, rownames(installed.packages()))
if (length(new)) install.packages(new)

library(ggplot2)
library(nnet)
library(randomForest)
options(repr.plot.width = 8, repr.plot.height = 4.5, repr.plot.res = 110)
theme_set(theme_minimal(base_size = 12))

REPO <- "abu85/ai-for-agri-bio"
data_path <- function(name) {
  for (folder in c("data", "../data", "../../data")) {
    f <- file.path(folder, name)
    if (file.exists(f)) return(f)
  }
  paste0("https://raw.githubusercontent.com/", REPO, "/main/data/", name)
}
rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))
set.seed(42)

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
sigmoid <- function(z) 1 / (1 + exp(-z))
neuron <- function(rainfall_mm, temp_c, w_rain = 0.006, w_temp = 0.35, bias = -8) {
  sigmoid(w_rain * rainfall_mm + w_temp * temp_c + bias)
}
for (x in list(c(300, 12), c(550, 15), c(750, 18))) {
  cat(sprintf("rainfall %d mm, %d °C -> disease risk %.2f\n", x[1], x[2], neuron(x[1], x[2])))
}

# %%
z <- seq(-6, 6, length.out = 200)
par(mfrow = c(1, 2))
plot(z, sigmoid(z), type = "l", lwd = 2, main = "Sigmoid: squashes to 0–1", xlab = "weighted sum z", ylab = "output")
plot(z, pmax(0, z), type = "l", lwd = 2, main = "ReLU: the default in deep networks", xlab = "weighted sum z", ylab = "output")
par(mfrow = c(1, 1))

# %% [markdown]
# A **network** connects many neurons in layers. Each hidden neuron learns its
# own combination of the inputs; the output neuron combines the hidden ones.
# Training adjusts all the weights, a little at a time, to reduce the
# prediction error. `nnet` uses an optimiser called BFGS; deep-learning
# libraries use gradient descent with backpropagation, one pass through the
# data per **epoch**. The idea is the same.
#
# ## 2. Train a network on field-trial data

# %%
trials <- read.csv(data_path("crop_trials.csv"))
trials$yield_t_ha[trials$yield_t_ha > 20] <- NA
trials$soil_ph[trials$soil_ph < 3] <- NA
trials <- trials[!is.na(trials$yield_t_ha), ]
features <- c("n_fert_kg_ha", "soil_p_mg_kg", "soil_k_mg_kg", "soil_ph", "organic_matter_pct",
              "rainfall_mm", "mean_temp_c", "irrigated", "variety")
X <- trials[, features]
for (col in names(X)) if (is.numeric(X[[col]])) X[[col]][is.na(X[[col]])] <- median(X[[col]], na.rm = TRUE)
X <- model.matrix(~ . - 1, data = X)  # one-hot encode variety
y <- trials$yield_t_ha

test_idx <- sample(nrow(X), round(0.25 * nrow(X)))
# Neural networks need inputs on similar scales: standardise to mean 0, SD 1,
# using means and SDs from the training data only.
mu <- colMeans(X[-test_idx, ]); sdv <- apply(X[-test_idx, ], 2, sd)
Xs <- scale(X, center = mu, scale = sdv)
X_train <- Xs[-test_idx, ]; X_test <- Xs[test_idx, ]
y_train <- y[-test_idx];   y_test <- y[test_idx]

# %% [markdown]
# We train in small steps (a few optimiser iterations at a time, continuing from
# the previous weights) and record the error on training and test data.

# %%
training_curve <- function(size, decay, steps = 60, iters_per_step = 5) {
  net <- nnet(X_train, y_train, size = size, decay = decay, linout = TRUE,
              maxit = iters_per_step, MaxNWts = 20000, trace = FALSE)
  curve <- data.frame()
  for (s in 1:steps) {
    net <- nnet(X_train, y_train, size = size, decay = decay, linout = TRUE, Wts = net$wts,
                maxit = iters_per_step, MaxNWts = 20000, trace = FALSE)
    curve <- rbind(curve, data.frame(iteration = s * iters_per_step,
                                     data = c("training data", "test data"),
                                     rmse = c(rmse(y_train, predict(net, X_train)),
                                              rmse(y_test, predict(net, X_test)))))
  }
  list(net = net, curve = curve)
}

res16 <- training_curve(size = 16, decay = 0.0001)
ggplot(res16$curve, aes(iteration, rmse, colour = data)) +
  geom_line() + coord_cartesian(ylim = c(0, 3)) +
  labs(title = "A network with 16 hidden neurons learning", x = "optimiser iterations", y = "RMSE (t/ha)", colour = NULL)

# %% [markdown]
# The error drops fast at first as the network finds the main patterns, then
# improves slowly.
#
# ## 3. Overfitting, and how to stop it
#
# A big network with no brakes can memorise the training data. Two common
# brakes are **weight decay** (L2 regularisation, `decay`: penalise large
# weights) and **early stopping** (stop when validation error stops improving).

# %%
big_free <- training_curve(size = 64, decay = 0)$curve
big_reg <- training_curve(size = 64, decay = 5)$curve
both <- rbind(cbind(big_free, model = "64 neurons, no regularisation"),
              cbind(big_reg, model = "64 neurons, strong weight decay"))
options(repr.plot.width = 10, repr.plot.height = 4)
ggplot(both, aes(iteration, rmse, colour = data)) +
  geom_line() + facet_wrap(~ model) + coord_cartesian(ylim = c(0, 3)) +
  labs(x = "optimiser iterations", y = "RMSE (t/ha)", colour = NULL)
options(repr.plot.width = 8, repr.plot.height = 4.5)

# %% [markdown]
# Without regularisation, training error keeps falling while test error stalls
# or creeps up: the network is memorising noise. With weight decay, the gap
# between training and test error stays small.
#
# ## 4. Compare with a random forest

# %%
net <- nnet(X_train, y_train, size = 16, decay = 1, linout = TRUE, maxit = 1000, MaxNWts = 20000, trace = FALSE)
forest <- randomForest(X[-test_idx, ], y_train, ntree = 500)
data.frame(model = c("baseline (mean)", "neural network (16 hidden)", "random forest"),
           test_RMSE_t_ha = round(c(rmse(y_test, mean(y_train)),
                                    rmse(y_test, predict(net, X_test)),
                                    rmse(y_test, predict(forest, X[test_idx, ]))), 3))

# %% [markdown]
# The network does well, but the random forest is as good or better — with no
# scaling and no training schedule to tune. **For tables of measurements,
# start with tree models.** Use deep learning when the input is an image, a
# sequence or text, where it can learn the features itself.
#
# **Try it:** Change `size` to 2 and to 128. When does the network underfit,
# and when does it overfit?
