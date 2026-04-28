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

# --- LOGGING SETUP ---
log_file_path = "VGG_With_Padding_Five_Log.txt"

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

# Redirect stdout so everything is logged
sys.stdout = DualLogger(log_file_path)

# --- Path Configuration ---
folder_with_data = '/workspace/data/komnet2' 
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

# Normalization & Optimization
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# --- Model Definition ---
def create_model(num_classes_arg, input_shape_arg=(256, 256, 3)):
    model = models.Sequential(name="VGG_Face_With_Padding")
    
   # Block 1: (256x256) -> (128x128)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape_arg, padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 2: (128x128) -> (64x64)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 3: (64x64) -> (32x32)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 4: (32x32) -> (16x16)
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 5: (16x16) -> (8x8)
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Classification Head
    # Final feature map is 8x8x512 = 32,768 neurons
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu')) 
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes_arg, activation='softmax'))
    
    return model

# --- Plotting Functions ---
def save_performance_plots(history, filename="VGG_With_Padding_Accuracy_Loss.png"):
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
    plt.title('VGG With Padding: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('VGG With Padding: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved accuracy/loss plots to {filename}")

def save_roc_plot(y_true, y_probs, num_classes, filename="VGG_With_Padding_ROC.png"):
    # Binarize labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute micro-average ROC curve
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Micro-average ROC (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('VGG With Padding: Multiclass ROC')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(filename)
    print(f"Saved ROC plot to {filename}")

# --- Custom Evaluation Function ---
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
    save_roc_plot(y_true, y_probs, len(class_names))
# --- Saliency Map / Variable Importance Function ---

def save_vgg_padded_saliency(model, dataset, class_names):
    # 1. Extract a sample image and its true label from the validation set
    for images, labels in dataset.take(1):
        img = images[0:1]  # Select the first image in the batch
        label_idx = labels[0].numpy()
        break

    # 2. Set up GradientTape to track the relationship between pixels and prediction
    img_tensor = tf.convert_to_tensor(img)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        
        # Target the class the model is most confident in
        top_class_index = tf.argmax(predictions[0])
        top_class_score = predictions[:, top_class_index]

    # 3. Calculate gradients: Which pixels most influence the final score?
    grads = tape.gradient(top_class_score, img_tensor)
    
    # Process gradients (absolute value and max across color channels)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    
    # Normalize for better visualization contrast
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)

    # 4. Create the Comparison Plot
    plt.figure(figsize=(14, 6))
    
    # Display Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img[0])
    plt.title(f"Original: {class_names[label_idx]}")
    plt.axis('off')

    # Display Saliency Map (Variable Importance)
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='viridis')
    plt.title("VGG Padded: Variable Importance")
    plt.colorbar(label='Influence Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('vgg_with_padding_variable_importance.png')
    print("Saved vgg_with_padding_variable_importance.png")



# --- Execution ---
model = create_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- MODEL ARCHITECTURE SUMMARY ---")
model.summary()

print("\nStarting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)

# Save Training Plots
save_performance_plots(history)

# Comprehensive Evaluation & ROC Plot
comprehensive_evaluation(model, val_ds, class_names)

# --- Final Execution ---
save_vgg_padded_saliency(model, val_ds, class_names)


print(f"\nAll results saved. Check {log_file_path}, VGG_With_Padding_Accuracy_Loss.png, and VGG_With_Padding_ROC.png")