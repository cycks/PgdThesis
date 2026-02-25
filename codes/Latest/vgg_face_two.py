import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, 
                                     Dropout, Flatten, Activation, Dense)
from sklearn.metrics import (classification_report, roc_curve,
                             auc, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Use 'Agg' backend to save plots without needing a GUI display
import matplotlib
matplotlib.use('Agg')

# --- LOGGING SETUP ---
log_file_path = "vgg_no_padding_performance_log.txt"
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"--- Starting execution and logging to '{log_file_path}' ---")

    # (Keep your existing data loading logic here: folder_with_zipped_data, train_ds, val_ds, test_ds, etc.)
    # ... [Assuming your data loading code from the prompt is here] ...

    # --- VGG-like Architecture ---
    def create_vgg_face_without_padding(input_shape=(256, 256, 3),
                                         num_classes_arg=None):
        if num_classes_arg is None: raise ValueError("num_classes_arg must be provided")
        model = models.Sequential(name="VGG_Face_without_Padding")
        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu',
                         input_shape=input_shape, padding='valid'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        # Flatten & Dense
        model.add(Flatten())
        model.add(Dense(1024, activation='relu')) # Reduced from 4096 for memory stability
        model.add(Dropout(0.5))
        model.add(Dense(num_classes_arg, activation='softmax'))
        return model

    # --- Modified Train and Evaluate Function ---
    def train_and_evaluate(model, verbose_setting=1):
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\n--- Model Summary ---")
        model.summary()

        print(f"\n--- Training {model.name} ---")
        start_time = time.time()
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=10, verbose=verbose_setting)
        
        duration = time.time() - start_time
        print(f"\nTraining completed in {duration // 60:.0f}m {duration % 60:.0f}s.")

        # --- ADVANCED EVALUATION ---
        print("\n" + "="*50)
        print("PERFORMANCE METRICS ON TEST DATA")
        print("="*50)

        y_true = []
        y_probs = [] # Softmax probabilities for ROC

        for images, labels in test_ds:
            probs = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_probs.extend(probs)

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = np.argmax(y_probs, axis=1)

        # 1. Standard Classification Report
        print("\n1. Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Vgg No Padding Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('Vgg no padding confusion_matrix.png')
        print("2. Confusion Matrix saved as 'confusion_matrix.png'")

        # 3. ROC Curve & AUC
        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(12, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Vgg No Padding Multiclass ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('Vgg no padding roc_curve.png')
        
        # Log Overall AUC
        macro_auc = roc_auc_score(y_true_bin, y_probs, 
                                  multi_class="ovr", average="macro")
        print(f"3. Macro-Average ROC-AUC Score: {macro_auc:.4f}")
        print("ROC Curve plot saved as 'roc_curve.png'")

    # Run execution
    model = create_vgg_face_without_padding(num_classes_arg=num_classes)
    train_and_evaluate(model)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}", file=original_stderr)
    raise
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Metrics logged to {log_file_path}. Images saved to directory.")