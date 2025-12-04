import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
TRAIN_DIR = 'mrl_data/Train' 
TEST_DIR = 'mrl_data/Test'
MODEL_NAME = 'models/mouth_model.h5'

# IMPORTANT: Now we target the mouth folders
TARGET_CLASSES = ['Yawn', 'No_yawn']

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32

def train_model():
    # 1. Create Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,        
        rotation_range=10,     
        zoom_range=0.1,        
        horizontal_flip=True   
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("\nLoading Training Data (Mouth)...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', 
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=TARGET_CLASSES # <--- Focusing on Mouths now
    )

    print("Loading Test Data (Mouth)...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=TARGET_CLASSES
    )

    # 2. Build Model
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        # Layer 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("\nStarting Training Mouth Model (5 Epochs)...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=5 
    )

    # 4. Save
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save(MODEL_NAME)
    print(f"\nSUCCESS! Model saved to {MODEL_NAME}")

    # Print accuracy
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()