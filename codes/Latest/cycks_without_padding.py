import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import label_binarize
import numpy as np
import time

# --- CRITICAL FOR VAST.AI: Non-interactive backend ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOGGING SETUP ---
log_file_path = "cycks_no_padding_log.txt"
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"--- Starting execution: {time.ctime()} ---")

    # Data Path & Config
    folder_with_zipped_data = './komnet'
    image_size = (256, 256)
    batch_size = 32

    # 1. Load Datasets
    rescale = tf.keras.Sequential([layers.Rescaling(1./255)])
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2, subset="training",
        seed=123, image_size=image_size, batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2, subset="validation",
        seed=123, image_size=image_size, batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

    # 2. Model Architecture
    def create_adjusted_cnn_model_valid_padding(input_shape=(256, 256, 3),
                                                num_classes=10):
        model = models.Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid',
                  input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    # 3. Plotting Functions
    def save_performance_plots(history, y_true, y_probs, class_names):
        # A. Learning Curves
        acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
        loss, val_loss = history.history['loss'], history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Acc')
        plt.plot(epochs_range, val_acc, label='Val Acc')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train Loss')
        plt.plot(epochs_range, val_loss, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig('cycks_no_padding_learning_curves.png')
        plt.close()

        # B. Confusion Matrix
        y_pred = np.argmax(y_probs, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Cycks No Padding Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('cycks_no_padding_confusion_matrix.png')
        plt.close()

        # C. Multi-class ROC Curve
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('Cycks No Padding Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('cycks_no_padding_roc_curve.png')
        plt.close()

    # 4. Training Loop
    def train_and_evaluate(model, epochs=10):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        start_time = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
        print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes.")

        # Final Predictions for Metrics
        y_true, y_probs = [], []
        for images, labels in val_ds:
            y_true.extend(labels.numpy())
            y_probs.extend(model.predict(images, verbose=0))
        
        y_true, y_probs = np.array(y_true), np.array(y_probs)
        
        # Log Text Metrics
        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_true, np.argmax(y_probs, axis=1), 
                                    target_names=class_names))
        
        macro_auc = roc_auc_score(label_binarize(y_true,
                                 classes=np.arange(num_classes)),
                                  y_probs, multi_class="ovr", average="macro")
        print(f"Macro-average ROC-AUC: {macro_auc:.4f}")

        # Save Visual Metrics
        save_performance_plots(history, y_true, y_probs, class_names)

    # Execution
    model = create_adjusted_cnn_model_valid_padding(num_classes=num_classes)
    train_and_evaluate(model, epochs=10)

except Exception as e:
    print(f"\nError: {e}", file=original_stderr)
    raise
finally:
    sys.stdout, sys.stderr = original_stdout, original_stderr
    log_file.close()
    print(f"Logging complete. Check '{log_file_path}' and generated .png files.")