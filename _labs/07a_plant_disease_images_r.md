::: {.callout-note}
This lab needs internet access (and ideally a GPU), so it is shown without outputs. Run it in Google Colab with the button above.
:::

**Run this lab in Google Colab with a GPU:** *Runtime → Change runtime type →
T4 GPU* (keep R as the language). It downloads the PlantVillage dataset
(about 830 MB) and a pre-trained network, so it needs internet access.

Deep learning in R uses the `keras3` package, which drives Python's
TensorFlow behind the scenes through `reticulate`. The first setup cell
installs that Python backend and can take 5–10 minutes. The model code itself
is plain R.

You will use **transfer learning**: take a network already trained on
millions of everyday photos (ImageNet), keep the layers that detect edges,
textures and shapes, and train only a new final layer to recognise leaf
diseases. The PlantVillage dataset has 54,303 leaf images in 38 classes.

```r
if (!requireNamespace("keras3", quietly = TRUE)) install.packages("keras3")
library(keras3)
library(reticulate)

# Install TensorFlow (Python) once per session, plus TensorFlow Datasets
if (!py_module_available("tensorflow")) install_keras(backend = "tensorflow")
if (!py_module_available("tensorflow_datasets")) py_install("tensorflow-datasets", pip = TRUE)

tf <- import("tensorflow")
tfds <- import("tensorflow_datasets")
cat("GPU available:", length(tf$config$list_physical_devices("GPU")) > 0, "\n")
set_random_seed(42)
```

### 1. Load the data

To keep the lab fast, we use 30% of the images: 20% for training and 10% for
testing.

```r
loaded <- tfds$load("plant_village", split = list("train[:20%]", "train[90%:]"),
                    as_supervised = TRUE, with_info = TRUE, shuffle_files = TRUE)
train_ds <- loaded[[1]][[1]]
test_ds <- loaded[[1]][[2]]
class_names <- loaded[[2]]$features[["label"]]$names
cat(length(class_names), "classes, e.g.", head(class_names, 4), "\n")
```

```r
par(mfrow = c(3, 4), mar = c(0.5, 0.5, 2, 0.5))
it <- as_iterator(train_ds$take(12L))
for (i in 1:12) {
  batch <- iter_next(it)
  img <- as.array(batch[[1]]) / 255
  plot(as.raster(img))
  title(gsub("___", "\n", class_names[as.integer(batch[[2]]) + 1]), cex.main = 0.7)
}
par(mfrow = c(1, 1))
```

Notice the uniform background and careful framing. Keep this in mind — it
matters at the end of the lab.

### 2. Prepare the images

The pre-trained network (MobileNetV2, small enough for phones) expects
224 × 224 pixel images with values scaled to −1…1.

```r
IMG <- 224L
# Written with TensorFlow operations so tf.data can run it on the GPU
prepare <- function(image, label) {
  image <- tf$image$resize(image, c(IMG, IMG))
  image <- tf$subtract(tf$divide(image, 127.5), 1)  # MobileNetV2 scaling to -1...1
  tuple(image, label)
}
train <- train_ds$map(prepare)$shuffle(1000L)$batch(32L)$prefetch(-1L)
test <- test_ds$map(prepare)$batch(32L)$prefetch(-1L)
```

### 3. Build the model: a frozen base and a new head

**Data augmentation** (random flips, rotations and zoom) shows the network
slightly different versions of each image, which reduces overfitting.

```r
base <- application_mobilenet_v2(input_shape = c(IMG, IMG, 3), include_top = FALSE, weights = "imagenet")
base$trainable <- FALSE  # keep the ImageNet features

inputs <- keras_input(shape = c(IMG, IMG, 3))
outputs <- inputs |>
  layer_random_flip("horizontal_and_vertical") |>
  layer_random_rotation(0.1) |>
  layer_random_zoom(0.1) |>
  base(training = FALSE) |>
  layer_global_average_pooling_2d() |>
  layer_dropout(0.2) |>
  layer_dense(length(class_names), activation = "softmax")
model <- keras_model(inputs, outputs)

model |> compile(optimizer = optimizer_adam(1e-3),
                 loss = "sparse_categorical_crossentropy",
                 metrics = "accuracy")
model
```

Only the final layer — about 49,000 of 2.3 million parameters — is trained.

### 4. Train

Three epochs take a few minutes on a Colab GPU.

```r
history <- model |> fit(train, validation_data = test, epochs = 3)
plot(history)
```

### 5. Where does it go wrong?

Overall accuracy hides which diseases get confused. List the most common
mistakes.

```r
y_true <- c(); y_pred <- c()
it <- as_iterator(test)
while (!is.null(batch <- iter_next(it, completed = NULL))) {
  y_true <- c(y_true, as.integer(batch[[2]]))
  y_pred <- c(y_pred, max.col(predict(model, batch[[1]], verbose = 0)) - 1L)
}
cat("test accuracy:", round(mean(y_true == y_pred), 3), "\n")
errors <- table(true = class_names[y_true + 1][y_true != y_pred],
                predicted = class_names[y_pred + 1][y_true != y_pred])
head(as.data.frame(errors)[order(-as.data.frame(errors)$Freq), ], 8)
```

### 6. Try your own photo

In Colab, upload a leaf photo with the file browser (folder icon on the left),
then set `photo` to its file name. Try a photo taken in a field, with soil,
other leaves or your hand in the background.

```r
photo <- "my_leaf.jpg"
if (file.exists(photo)) {
  img <- image_load(photo, target_size = c(IMG, IMG))
  arr <- array_reshape(image_to_array(img), c(1, IMG, IMG, 3)) / 127.5 - 1
  probs <- predict(model, arr, verbose = 0)[1, ]
  top <- order(probs, decreasing = TRUE)[1:3]
  data.frame(class = class_names[top], probability = round(probs[top], 2))
} else {
  cat("Upload a photo and set `photo` to its file name.\n")
}
```

### The catch: lab photos are not field photos

Models trained on PlantVillage score very high on PlantVillage test images. In
the original study (Mohanty, Hughes and Salathé, 2016, *Frontiers in Plant
Science*), a network that reached 99.35% on held-out PlantVillage images got
31.4% right on leaf images collected from other sources under different
conditions. The model partly learned the photo setup, not only the disease.

**Discussion**

- What would you need to collect to build a model that works in *your* fields?
- How would you test it honestly? (Think of Lab 04: new farms, new seasons,
  new phones.)
- What should an app do when it is unsure, or sees a crop it was never trained on?
