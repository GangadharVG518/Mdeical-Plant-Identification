import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_alternative_leaf_model():
    """Builds a CNN model for alternative leaf classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Adjust based on alternative leaf categories
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_alternative_leaf_model():
    """Train alternative leaf model and save it as leaf.h5."""
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        "dataset/alternative_leaf", target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
    validation_generator = datagen.flow_from_directory(
        "dataset/alternative_leaf", target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

    model = build_alternative_leaf_model()
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save("leaf.h5")
    print("Alternative leaf model saved as leaf.h5")

if __name__ == "__main__":
    train_alternative_leaf_model()
