import zipfile
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                        Dropout, BatchNormalization, GlobalAveragePooling2D)
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, balanced_accuracy_score, confusion_matrix, roc_curve, auc
)
import numpy as np
from itertools import cycle

# --- LOGGING SETUP ---
log_file_path = "Reduced_cycks_With_Padding_1_Log.txt"

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

sys.stdout = DualLogger(log_file_path)

# --- Path Configuration ---
folder_with_zipped_data = '/workspace/data/komnet2'

# --- Data Loading ---
image_size = (256, 256)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    folder_with_zipped_data,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    folder_with_zipped_data,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Normalization & Optimization
rescale = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y)).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (rescale(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# --- Model Definition ---
def create_model(num_classes, input_shape=(256, 256, 3)):
    model = models.Sequential(name="Cycks_With_Padding")
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# --- Plotting Functions ---
def save_performance_plots(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy for Cycks With Padding')
    plt.legend(loc='lower right')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    
    plt.savefig('Reduced_cycks_with_padding_1_accuracy_loss_plots.png')
    print("Saved Reduced_cycks_with_padding_1_accuracy_loss_plots.png")

def save_roc_curve(y_true, y_probs, num_classes):
    # For multiclass, we use One-vs-Rest
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve for Cycks With Padding')
    plt.legend(loc="lower right")
    plt.savefig('Reduced_cycks_with_padding_1_roc_curve.png')
    print("Saved roc_curve.png")

# --- Evaluation Function ---
def comprehensive_evaluation(model, dataset, class_names):
    print("\n" + "="*40)
    print("      DETAILED PERFORMANCE METRICS")
    print("="*40)
    
    y_true = []
    y_probs = []
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        y_probs.append(model.predict(images, verbose=0))
    
    y_true = np.array(y_true)
    y_probs = np.vstack(y_probs)
    y_pred = np.argmax(y_probs, axis=1)

    # Calculate metrics
    acc = np.mean(y_true == y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    loss_val = log_loss(y_true, y_probs)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')

    print(f"Accuracy: {acc:.4f} | Balanced Acc: {b_acc:.4f} | Log Loss: {loss_val:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save ROC plot
    save_roc_curve(y_true, y_probs, len(class_names))

# --- Execution ---
model = create_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- MODEL ARCHITECTURE SUMMARY ---")
model.summary()

print("\nStarting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)

# Save Accuracy/Loss plots
save_performance_plots(history)

# Run evaluation and save ROC plot
comprehensive_evaluation(model, val_ds, class_names)