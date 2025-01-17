import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW
from speak_spell import play_spell_sound

def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padding = [[0.0, 0.0] for _ in range(max_length - len(seq))]
        padded_sequences.append(seq + padding)
    return np.array(padded_sequences)


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
        normalized_xy = [(xy[0]-min_xy[0]) / (max_xy[0]-min_xy[0]), (xy[1]-min_xy[1])/(max_xy[1]-min_xy[1])]
        normalized_seq.append(normalized_xy)
      normalized_sequences.append(normalized_seq)
    return normalized_sequences

def augment_sequences(sequences, noise_factor=0.01):
    augmented_sequences = []
    for sequence in sequences: # Added this for statement to handle the lists of lists
      augmented_seq = []
      for xy in sequence:
          noise_x = np.random.normal(0, noise_factor)
          noise_y = np.random.normal(0, noise_factor)
          augmented_xy = [xy[0] + noise_x, xy[1] + noise_y]
          augmented_seq.append(augmented_xy)
      augmented_sequences.append(augmented_seq)
    return augmented_sequences

# 6. Inference
def predict_path_group(new_path, model):
  #pre-process new path
    padded_new_path = pad_sequences([new_path])
    normalized_new_path = normalize_sequences([new_path])
    augmented_new_path = augment_sequences(normalized_new_path, noise_factor=0.01) # normalized_new_path wrapped inside a list
    padded_normalized_new_path = pad_sequences(augmented_new_path)

    probabilities = model.predict(padded_normalized_new_path)
    data = np.array(probabilities)

    # Convert to binary: highest value is 1, others are 0
    binary_data = (data == np.max(data)).astype(int)

    predicted_group = np.argmax(probabilities, axis=1)[0]
    
    return [predicted_group,binary_data]




def preds(model_path, csv_path):
    loaded_model = tf.keras.models.load_model(model_path)
    example_path_dataframe = pd.read_csv(csv_path) #replace with an actual test file
    example_path = example_path_dataframe[['x', 'y']].values.tolist()
    prediction = predict_path_group(example_path, loaded_model)
    binary_pred = prediction[1]
    prediction = prediction[0]
    # print(example_path)
    print(binary_pred)
    print(f"Predicted path group: {int(prediction) + 1}")
    play_spell_sound(int(prediction) + 1)
    

# preds("path_matching_model.keras","./data/compiled/1/4.csv")