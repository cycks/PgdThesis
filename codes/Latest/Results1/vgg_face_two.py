import sys
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, Dense)
from sklearn.metrics import (classification_report, roc_curve, auc, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# --- HEADLESS PLOTTING SETUP ---
import matplotlib
matplotlib.use('Agg')

# --- LOGGING SETUP ---
log_file_path = "vgg_execution_log.txt"
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    log_file = open(log_file_path, 'w', encoding='utf-8')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"--- Starting execution: {time.ctime()} ---")

    # --- 1. DATA HANDLING ---
    zip_filename = 'komnet.zip'
    extraction_dir = './shared_komnet_data'

    # Logic: Only extract if the shared folder doesn't already exist
    if not os.path.exists(extraction_dir):
        if os.path.exists(zip_filename):
            print(f"Found {zip_filename}. Extracting to {extraction_dir}...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            print(f"Extraction complete.")
        else:
            print(f"CRITICAL ERROR: {zip_filename} not found and {extraction_dir} is missing.")
            sys.exit(1)
    else:
        print(f"Using existing shared directory: {extraction_dir}")

    # --- 2. DATA LOADING ---
    image_size = (256, 256)
    batch_size = 32

    # Load datasets
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        extraction_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        extraction_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)
    print(f"Classes detected: {class_names}")

    # Create test set from training data
    train_batches = tf.data.experimental.cardinality(train_ds_raw).numpy()
    test_ds = train_ds_raw.take(max(1, train_batches // 5))
    train_ds = train_ds_raw.skip(max(1, train_batches // 5))

    # Preprocessing pipeline
    rescale = layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)
    val_ds = val_ds_raw.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(AUTOTUNE)

    # --- 3. ARCHITECTURE: VGG WITHOUT PADDING ---
    def create_vgg_no_padding(input_shape=(256, 256, 3), num_classes_arg=num_classes):
        model = models.Sequential(name="VGG_No_Padding")
        
        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='valid'))
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

    # --- 4. TRAINING ---
    model = create_vgg_no_padding()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    print(f"\n--- Training {model.name} ---")
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration // 60:.0f}m {duration % 60:.0f}s.")

    # --- 5. EVALUATION ---
    y_true, y_probs = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_probs.extend(probs)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.argmax(y_probs, axis=1)

    print("\n1. Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('VGG (No Padding) Confusion Matrix')
    plt.savefig('vgg_no_padding_confusion_matrix.png')

    # ROC Curve
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('VGG No Padding Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('vgg_no_padding_roc_curve.png')
    
    macro_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr", average="macro")
    print(f"\n3. Macro-Average ROC-AUC Score: {macro_auc:.4f}")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}", file=original_stderr)
    raise
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
    print(f"Process finished. Results logged to {log_file_path}")