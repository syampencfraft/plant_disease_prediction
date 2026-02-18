import os
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from detection.cnn_algorithm import create_cnn_model

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = 'dataset/PlantVillage-Dataset-master/raw/color' # Path inside the extracted archive

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found. Run download_dataset.py first.")
        return

    # Data Augmentation for high accuracy
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    model = create_cnn_model(num_classes=num_classes)

    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stop]
    )

    model.save('plant_disease_model.keras')
    print("Model saved to plant_disease_model.keras")

    # Save class indices mapping
    class_indices = train_generator.class_indices
    # Reverse the mapping to be {index: name}
    class_mapping = {v: k for k, v in class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_mapping, f)
    print("Class mapping saved to class_indices.json")

if __name__ == "__main__":
    train()
