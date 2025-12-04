import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
# matching the capitalization in your screenshots
TRAIN_DIR = 'mrl_data/Train' 
TEST_DIR = 'mrl_data/Test'
MODEL_NAME = 'models/eye_model.h5'

# We explicitly tell the model which folders to use.
# It will ignore "Yawn" and "No_yawn" automatically.
TARGET_CLASSES = ['Open_Eyes', 'Closed_Eyes']

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

    print("\nLoading Training Data...")
    # We add classes=TARGET_CLASSES to filter the folders
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', 
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=TARGET_CLASSES  # <--- This is the magic fix
    )

    print("Loading Test Data...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=TARGET_CLASSES # <--- This is the magic fix
    )

    # 2. Build Model
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("\nStarting Training (5 Epochs)...")
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