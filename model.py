from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Load Dataset
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'   # CHANGED
)
print("Class indices:", train_data.class_indices)


val_data = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'   # CHANGED
)

num_classes = train_data.num_classes  # Should be 4

# -----------------------------
# Build CNN Model
# -----------------------------
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # CHANGED: 4 outputs
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # CHANGED
        metrics=['accuracy']
    )

    return model

# -----------------------------
# Train & Save
# -----------------------------
model = build_model()
model.summary()

model.fit(train_data, epochs=10, validation_data=val_data)
model.save("lung_cancer_model_4class.h5")

print("4-class model saved successfully!")
