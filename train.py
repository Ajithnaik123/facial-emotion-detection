import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Configuration
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_SAVE_PATH = "models/emotion_model.keras"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30

os.makedirs("models", exist_ok=True)

# 2. Load Datasets (48x48 Grayscale, 7 Categorical Classes)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
print("Detected Classes:", class_names)

# Optimize input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 3. Build CNN Architecture
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Rescaling(1.0 / 255),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Classification Head
    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

# 4. Learning Rate Schedule & Compilation
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=100000,
    decay_rate=0.96
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train & Save Model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")