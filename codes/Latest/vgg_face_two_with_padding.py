import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import time

# --- ADDED: Backend for non-GUI environments ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOGGING SETUP ---
log_file_path = "vgg_with_padding_log.txt"
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"--- Starting execution: {time.ctime()} ---")

    # [Data Loading Logic - Keep your existing block here]
    folder_with_zipped_data = './komnet2'
    image_size = (256, 256)
    batch_size = 32
    rescale = tf.keras.Sequential([layers.Rescaling(1./255)])

    initial_train_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2, subset="training", seed=123, image_size=image_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder_with_zipped_data, validation_split=0.2, subset="validation", seed=123, image_size=image_size, batch_size=batch_size
    )

    class_names = initial_train_ds.class_names
    num_classes = len(class_names)
    
    # Split initial_train into train and test
    train_batches_count = tf.data.experimental.cardinality(initial_train_ds).numpy()
    test_size_batches = max(1, train_batches_count // 4)
    test_ds = initial_train_ds.take(test_size_batches)
    train_ds = initial_train_ds.skip(test_size_batches)

    # Prefetch and Map
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(buffer_size=AUTOTUNE)

    def create_vgg_face_with_padding(input_shape=(256, 256, 3), num_classes_arg=num_classes):
        model = models.Sequential(name="VGG_Face_With_Padding")
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu')) # Reduced size slightly for memory stability
        model.add(Dropout(0.5))
        model.add(Dense(num_classes_arg, activation='softmax'))
        return model

    def train_and_evaluate(model, verbose_setting=1):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print(f"\n--- Training {model.name} ---")
        start_time = time.time()
        model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=verbose_setting)
        print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")

        # --- ADVANCED EVALUATION ---
        y_true = []
        y_probs = [] # Raw probabilities for ROC

        print("\nCollecting predictions for advanced metrics...")
        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            y_probs.extend(model.predict(images, verbose=0))

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = np.argmax(y_probs, axis=1)

        # 1. Classification Report
        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # 2. Confusion Matrix
        print("\nGenerating Confusion Matrix...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Vgg with Padding Confusion Matrix: {model.name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('Vgg_with_padding_confusion_matrix.png')
        plt.close()

        # 3. ROC Curve
        print("Generating ROC Curves...")
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        plt.figure(figsize=(10, 8))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Vgg with Padding Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('Vgg_with_padding_roc_curve.png')
        plt.close()
        
        macro_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro")
        print(f"Macro-Average ROC-AUC: {macro_auc:.4f}")

    # Run execution
    model_vgg = create_vgg_face_with_padding()
    train_and_evaluate(model_vgg)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}", file=original_stderr)
    raise
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Logging complete. Results in {log_file_path}")

