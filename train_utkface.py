import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

# Path to your UTKFace folder
DATA_DIR = 'data/UTKFace'

# Desired image size (width, height)
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS = 10

def parse_utkface_filenames(data_dir):
    """
    Scans the UTKFace folder, extracts the age from each file name,
    and returns a DataFrame with 'image_path' and 'age' columns.
    """
    all_files = os.listdir(data_dir)
    image_paths = []
    ages = []

    for filename in all_files:
        if not filename.lower().endswith('.jpg'):
            continue

        splitted = filename.split('_')
        try:
            age = int(splitted[0])
        except ValueError:
            continue

        full_path = os.path.join(data_dir, filename)
        image_paths.append(full_path)
        ages.append(age)

    df = pd.DataFrame({'image_path': image_paths, 'age': ages})
    return df

def create_dataset(df):
    """
    Converts a DataFrame of image paths and ages into a tf.data.Dataset.
    The dataset yields (image_tensor, age) pairs.
    """
    file_paths = df['image_path'].values
    labels = df['age'].values

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def preprocess(file_path, label):
    """
    Reads an image, preprocesses it, and returns (img_tensor, label).
    """
    [img, label] = tf.py_function(load_and_preprocess_image, [file_path, label],
                                  [tf.float32, tf.int64])
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    label.set_shape(())
    label = tf.cast(label, tf.float32)
    return img, label

def load_and_preprocess_image(file_path, label):
    """
    Loads an image using OpenCV, converts to RGB, resizes, and normalizes.
    Returns (image_tensor, label).
    """
    file_path_str = file_path.numpy().decode('utf-8')
    img = cv2.imread(file_path_str)

    if img is None:
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    img = img.astype(np.float32) / 255.0
    return img, label

def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Creates a simple CNN for age regression.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Regression output
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])
    return model

if __name__ == '__main__':
    # Step 1: Parse UTKFace folder
    df = parse_utkface_filenames(DATA_DIR)
    print(f"Total images found: {len(df)}")
    print(df.head())

    # Step 2: Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Step 3: Create tf.data datasets
    train_ds = create_dataset(train_df)
    val_ds = create_dataset(val_df)

    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Step 4: Build and train model
    model = build_model()
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds, 
        epochs=EPOCHS
    )

    # Step 5: Save the trained model
    model.save("age_model.h5")
    print("Model saved as 'age_model.h5'.")
