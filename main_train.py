import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import glob
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW

# 1. Load Data and Preprocess
def load_and_preprocess_data(data_dir):
    all_sequences = []
    all_labels = []
    for group_num in range(1, 26):  # Iterate through groups 1 to 25
        group_dir = os.path.join(data_dir, str(group_num))
        if os.path.exists(group_dir):
            for csv_file in glob.glob(os.path.join(group_dir, '*.csv')):
                df = pd.read_csv(csv_file)
                coords = df[['x', 'y']].values.tolist()
                all_sequences.append(coords)
                all_labels.append(group_num - 1)  # Labels are from 0 to 24

    return all_sequences, all_labels

# Path to your compiled data
data_dir = "Final Data"  # Update accordingly
all_sequences, all_labels = load_and_preprocess_data(data_dir)

# Normalize Sequences
def normalize_sequences(sequences):
    # Convert to numpy array to easily find min and max
    sequences_np = np.concatenate(sequences, axis=0)

    # Find min and max values
    min_xy = np.min(sequences_np, axis=0)
    max_xy = np.max(sequences_np, axis=0)

    # Normalize each point in the sequences
    normalized_sequences = []
    for seq in sequences:
        normalized_seq = []
        for xy in seq:
            normalized_xy = [
                (xy[0] - min_xy[0]) / (max_xy[0] - min_xy[0]),
                (xy[1] - min_xy[1]) / (max_xy[1] - min_xy[1]),
            ]
            normalized_seq.append(normalized_xy)
        normalized_sequences.append(normalized_seq)
    return normalized_sequences

all_sequences = normalize_sequences(all_sequences)

# Data Augmentation
def augment_sequences(sequences, labels, num_augmentations=5, noise_factor=0.01, is_training=True):
    augmented_sequences = []
    augmented_labels = []
    for i, sequence in enumerate(sequences):  # Iterate with index
        # Original sequence with noise
        if is_training:
            augmented_seq = []
            for xy in sequence:
              noise_x = np.random.normal(0, noise_factor)
              noise_y = np.random.normal(0, noise_factor)
              augmented_xy = [xy[0] + noise_x, xy[1] + noise_y]
              augmented_seq.append(augmented_xy)
            augmented_sequences.append(augmented_seq)
            augmented_labels.append(labels[i])
        else:
            augmented_sequences.append(sequence)
            augmented_labels.append(labels[i])
        # Additional scaling augmentations
        for _ in range(num_augmentations):
            scale_factor = np.random.uniform(0.7, 1.3) # Scale between 0.7 and 1.3
            scaled_seq = []
            
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            for xy in sequence:
                scaled_x = xy[0] * scale_factor
                scaled_y = xy[1] * scale_factor
                
                min_x = min(min_x, scaled_x)
                min_y = min(min_y, scaled_y)
                max_x = max(max_x, scaled_x)
                max_y = max(max_y, scaled_y)
                scaled_seq.append([scaled_x, scaled_y])
                
            # Check if any points go out of bounds, and if so, skip augmentation
            if min_x < 0 or min_y < 0 or max_x > 1 or max_y > 1:
                continue

            # Translate the shape back into the [0,1] bounding box
            translation_x = 0
            translation_y = 0
            if min_x < 0:
                translation_x = -min_x
            if min_y < 0:
                translation_y = -min_y

            translated_scaled_seq = []
            for xy in scaled_seq:
              translated_scaled_seq.append([xy[0] + translation_x, xy[1] + translation_y])

            
            
            augmented_sequences.append(translated_scaled_seq)
            augmented_labels.append(labels[i]) # Added this to also duplicate the labels
    return augmented_sequences, augmented_labels

all_sequences, all_labels = augment_sequences(all_sequences, all_labels, is_training=True) # Modified to handle labels and indicate training

# Pad Sequences
def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padding = [[0.0, 0.0] for _ in range(max_length - len(seq))]
        padded_sequences.append(seq + padding)
    return np.array(padded_sequences)

X = pad_sequences(all_sequences)
y = np.array(all_labels)

# Split the data into training (70%) and temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the temporary data into validation (20%) and testing (10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Check dataset sizes
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")

# 2. Define Model
def create_model(sequence_length, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, 2)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128, return_sequences=True),  # Added second LSTM layer
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = AdamW(learning_rate=0.0005, weight_decay=0.004)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

max_sequence_length = X.shape[1]
num_classes = 25  # Number of classes/groups
model = create_model(max_sequence_length, num_classes)
model.summary()

# 3. Train Model
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
)  # Adjust factor and patience
model.fit(
    X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val),
    callbacks=[lr_scheduler]
)

# 4. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# 5. Save Model
model.save('path_matching_model556699.keras')

# 6. Inference
def predict_path_group(new_path, model):
    # Pre-process new path
    padded_new_path = pad_sequences([new_path])
    normalized_new_path = normalize_sequences([new_path])
    augmented_new_path, _ = augment_sequences([normalized_new_path], [0], noise_factor=0.01, is_training=False)  # normalized_new_path wrapped inside a list
    padded_normalized_new_path = pad_sequences(augmented_new_path)
    probabilities = model.predict(padded_normalized_new_path)
    predicted_group = np.argmax(probabilities, axis=1)[0]
    return predicted_group

loaded_model = tf.keras.models.load_model('path_matching_model556699.keras')
example_path_dataframe = pd.read_csv("./data/compiled/1/4.csv")  # Replace with an actual test file
example_path = example_path_dataframe[['x', 'y']].values.tolist()
prediction = predict_path_group(example_path, loaded_model)
print(f"Predicted path group: {prediction}")