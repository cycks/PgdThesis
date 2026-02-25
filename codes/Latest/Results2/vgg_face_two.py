import zipfile
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, log_loss, balanced_accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# --- LOGGING SYSTEM ---
log_file_path = "VGG_No_Padding_Log.txt"

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to the dual logger
sys.stdout = DualLogger(log_file_path)

# --- Configuration ---
folder_with_data = '/workspace/extracted_data/komnet2' 
image_size = (256, 256)
batch_size = 32

# --- Load datasets ---
print(f"Loading datasets from: {folder_with_data}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    folder_with_data,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    folder_with_data,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Normalization
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# --- Model Definition ---
def create_model(num_classes_arg, input_shape_arg=(256, 256, 3)):
    model = models.Sequential(name="VGG_No_Padding")
    # Block 1: Valid padding means shrinkage
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape_arg, padding='valid'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(num_classes_arg, activation='softmax'))
    return model

# --- Plotting Functions ---
def save_training_plots(history, acc_filename="VGG_No_Padding_Accuracy_Loss.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('VGG No Padding: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('VGG No Padding: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(acc_filename)
    print(f"Saved performance plots to {acc_filename}")

def save_roc_plot(y_true, y_probs, num_classes, roc_filename="VGG_No_Padding_ROC.png"):
    # Binarize labels for One-vs-Rest ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute micro-average ROC curve
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('VGG No Padding: Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(roc_filename)
    print(f"Saved ROC plot to {roc_filename}")

# --- Custom Evaluation Function ---
def comprehensive_evaluation(model, dataset, class_names):
    print("\n" + "="*40)
    print("      FINAL PERFORMANCE METRICS")
    print("="*40)
    
    y_true, y_probs = [], []
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        y_probs.append(model.predict(images, verbose=0))
    
    y_probs = np.vstack(y_probs)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(y_true)

    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC (Weighted): {roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted'):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save the ROC curve
    save_roc_plot(y_true, y_probs, len(class_names))

# --- Execution ---
model = create_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- MODEL ARCHITECTURE SUMMARY ---")
model.summary()

print("\nStarting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)

# Save the learning curves (Accuracy & Loss)
save_training_plots(history)

# Run evaluation and save ROC plot
comprehensive_evaluation(model, val_ds, class_names)

print(f"\nTraining complete. Files saved: {log_file_path}, VGG_No_Padding_Accuracy_Loss.png, VGG_No_Padding_ROC.png")