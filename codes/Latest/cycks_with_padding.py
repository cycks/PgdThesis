import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg') # Essential for remote servers/Vast.ai
import matplotlib.pyplot as plt
import seaborn as sns # For better confusion matrix visualization
import numpy as np
import time

# --- INITIAL SETUP ---
log_file_path = "cycks_with_padding_log.txt"
roc_plot_path = "roc_curve_plot_cycks_padding.png"
cm_plot_path = "confusion_matrix_cycks_padding.png"

def create_adjusted_cnn_model_same_padding(input_shape=(256, 256, 3), num_classes=10):
    model = models.Sequential()
    # Adding multiple blocks with 'same' padding to preserve spatial info
    for filters in [32, 64, 128, 256, 512]:
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same', input_shape=input_shape if filters==32 else None))
        model.add(BatchNormalization())
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# --- Data Parameters ---
folder_with_zipped_data = './komnet'
image_size = (256, 256)
batch_size = 32

# --- START LOGGING ---
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"--- Starting execution: {time.ctime()} ---")

    # Load Full Datasets
    rescale = tf.keras.Sequential([layers.Rescaling(1./255)])
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2,
        subset="training", seed=123, image_size=image_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2,
        subset="validation", seed=123,
        image_size=image_size, batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

    def train_and_evaluate(model, epochs=10):
        model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        
        print(f"\nTraining for {epochs} epochs...")
        start_time = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
        print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes.")

        # --- METRICS CALCULATION ---
        y_true = []
        y_probs = []

        print("\nCollecting predictions...")
        for images, labels in val_ds:
            y_true.extend(labels.numpy())
            preds = model.predict(images, verbose=0)
            y_probs.extend(preds)

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = np.argmax(y_probs, axis=1)

        # 1. Classification Report
        print("\n" + "="*30 + "\nClassification Report:\n" + "="*30)
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 2. Confusion Matrix Plot
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Cycks Padding Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion Matrix saved to: {cm_plot_path}")

        # 3. ROC-AUC Calculation
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        macro_roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr",
                                      average="macro")
        print(f"Macro-average ROC-AUC: {macro_roc_auc:.4f}")

        # 4. ROC Curve Plot
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Cycks Padding Multiclass ROC Curve')
        plt.legend(loc="lower right", fontsize='x-small', ncol=2)
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"ROC Curve plot saved to: {roc_plot_path}")

    model_same_padding = create_adjusted_cnn_model_same_padding(num_classes=num_classes)
    train_and_evaluate(model_same_padding, epochs=10)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}", file=original_stderr)
    raise
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Execution finished. Check '{log_file_path}' for logs.")