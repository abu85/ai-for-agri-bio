# %% [markdown]
# # Lab 07A — Recognising plant diseases from leaf images
#
# **Run this lab in Google Colab with a GPU:** *Runtime → Change runtime type →
# T4 GPU*. It downloads the PlantVillage dataset (about 830 MB) and a
# pre-trained network, so it needs internet access.
#
# You will use **transfer learning**: take a network already trained on
# millions of everyday photos (ImageNet), keep the layers that detect edges,
# textures and shapes, and train only a new final layer to recognise leaf
# diseases. This works with far fewer images and much less computing time than
# training from scratch.
#
# The PlantVillage dataset has 54,303 leaf images in 38 classes (14 crops,
# healthy or with one of 26 diseases).

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

print("TensorFlow", tf.__version__, "| GPU available:", bool(tf.config.list_physical_devices("GPU")))
SEED = 42
keras.utils.set_random_seed(SEED)

# %% [markdown]
# ## 1. Load the data
#
# To keep the lab fast, we use 30% of the images: 20% for training and 10% for
# testing. Increase these slices if you have time.

# %%
(train_ds, test_ds), info = tfds.load(
    "plant_village",
    split=["train[:20%]", "train[90%:]"],
    as_supervised=True,
    with_info=True,
    shuffle_files=True,
)
class_names = info.features["label"].names
print(f"{len(class_names)} classes, e.g. {class_names[:4]}")

# %%
plt.figure(figsize=(12, 6))
for i, (image, label) in enumerate(train_ds.take(12)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(class_names[label].replace("___", "\n"), fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# Notice the uniform background and careful framing. Keep this in mind — it
# matters at the end of the lab.
#
# ## 2. Prepare the images
#
# The pre-trained network (MobileNetV2, small enough for phones) expects
# 224 × 224 pixel images with values scaled to −1…1.

# %%
IMG = 224
BATCH = 32


def prepare(image, label):
    image = tf.image.resize(image, (IMG, IMG))
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


train = train_ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(BATCH).prefetch(tf.data.AUTOTUNE)
test = test_ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# ## 3. Build the model: a frozen base and a new head
#
# **Data augmentation** (random flips, rotations and zoom) shows the network
# slightly different versions of each image, which reduces overfitting.

# %%
base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False  # keep the ImageNet features

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

inputs = keras.Input(shape=(IMG, IMG, 3))
x = augment(inputs)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
print(f"total parameters: {model.count_params():,}; trained here: {trainable:,}")

# %% [markdown]
# Only the final layer — about 49,000 of 2.3 million parameters — is trained.
#
# ## 4. Train
#
# Three epochs take a few minutes on a Colab GPU.

# %%
history = model.fit(train, validation_data=test, epochs=3)

# %%
plt.plot(history.history["accuracy"], marker="o", label="training")
plt.plot(history.history["val_accuracy"], marker="o", label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Transfer learning on PlantVillage")
plt.show()

# %% [markdown]
# ## 5. Where does it go wrong?
#
# Overall accuracy hides which diseases get confused. List the most common
# mistakes.

# %%
y_true, y_pred = [], []
for images, labels in test:
    y_true.extend(labels.numpy())
    y_pred.extend(model.predict(images, verbose=0).argmax(axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)
print(f"test accuracy: {(y_true == y_pred).mean():.3f}")

pairs = {}
for t, p in zip(y_true, y_pred):
    if t != p:
        pairs[(class_names[t], class_names[p])] = pairs.get((class_names[t], class_names[p]), 0) + 1
for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]:
    print(f"{n:3d} x  true {t:45s} predicted {p}")

# %% [markdown]
# ## 6. Try your own photo
#
# Upload a leaf photo (in Colab) and see what the model says. Try a photo taken
# in a field, with soil, other leaves or your hand in the background.

# %%
try:
    from google.colab import files

    uploaded = files.upload()
    for name in uploaded:
        img = keras.utils.load_img(name, target_size=(IMG, IMG))
        arr = keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype="float32")[None])
        probs = model.predict(arr, verbose=0)[0]
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        for i in probs.argsort()[::-1][:3]:
            print(f"{probs[i]:.2f}  {class_names[i]}")
except ImportError:
    print("Photo upload works in Google Colab.")

# %% [markdown]
# ## The catch: lab photos are not field photos
#
# Models trained on PlantVillage score very high on PlantVillage test images. In
# the original study (Mohanty, Hughes and Salathé, 2016, *Frontiers in Plant
# Science*), a network that reached 99.35% on held-out PlantVillage images got
# 31.4% right on leaf images collected from other sources under different
# conditions. The model partly learned the photo setup, not only the disease.
#
# **Discussion**
#
# - What would you need to collect to build a model that works in *your* fields?
# - How would you test it honestly? (Think of Lab 04: new farms, new seasons,
#   new phones.)
# - What should an app do when it is unsure, or sees a crop it was never trained on?
#
# **Try it:** Unfreeze the top 30 layers of `base` (`base.trainable = True`, then
# set `layer.trainable = False` for all but the last 30), recompile with a 10×
# smaller learning rate, and train 2 more epochs. Does fine-tuning help?
