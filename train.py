import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, RandomRotation, RandomTranslation,
    RandomZoom, GaussianNoise
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

IMAGE_SIZE = 48
ALL_DATA_DIR = './images'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
RANDOM_SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
TEST_SPLIT_RATIO = 0.1
MODEL_SAVE_PATH = 'emotion_cnn_model_v3.h5'
VALIDATION = False


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_emotion_data(data_dir, image_size):

    data_list = []
    labels_list = []

    required_shape = (image_size, image_size)

    discarded_count = 0

    print(f"ladowanie danych z: {data_dir}")


    for emotion_label in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion_label)

        if os.path.isdir(emotion_path):
            for image_filename in os.listdir(emotion_path):
                if image_filename.lower().endswith(SUPPORTED_FORMATS):
                    image_path = os.path.join(emotion_path, image_filename)

                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


                    if image is None or image.ndim != 2 or image.shape != required_shape:
                        discarded_count += 1
                        continue

                    data_list.append(image)
                    labels_list.append(emotion_label)

    print(f"\nzaladowano {len(data_list)} obrazow.")
    print(f"liczba pominietych plikow: {discarded_count}")

    return data_list, labels_list

def create_augmentation_layers():

    return Sequential([
        RandomRotation(factor=0.1, fill_mode='reflect', seed=RANDOM_SEED),
        RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect', seed=RANDOM_SEED),
        RandomZoom(height_factor=0.1, width_factor=0.1, seed=RANDOM_SEED),
        GaussianNoise(0.05)
    ], name="data_augmentation")

def visualize_augmentation_effect(X_data, augmentation_model, num_samples=5):

    plt.figure(figsize=(12, 4 * num_samples))

    indices = np.random.choice(len(X_data), num_samples, replace=False)

    for i, idx in enumerate(indices):
        original_img = X_data[idx]

        img_batch = np.expand_dims(original_img, axis=0)

        augmented_img = augmentation_model(img_batch, training=True)[0].numpy()

        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(original_img.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        plt.title(f"Oryginał (Index: {idx})")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(augmented_img.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        plt.title("Po augmentacji")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def data_summary(labels):
    if labels:
        labels_np = np.array(labels)
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        emotion_counts = dict(zip(unique_labels, counts))

        print(f"\nliczba etykiet: {labels_np.shape[0]}")
        print("liczebnosc dla kazdej emocji:")

        for label, count in emotion_counts.items():
            print(f"  - {label:<10}: {count} obrazow")

        print(f"\ncalkowita liczba klas: {len(unique_labels)}")
    else:
        print(f"\nbrak danych do przetworzenia.")


def visualize_data_balance(labels, title_suffix):
    labels_np = np.array(labels)
    unique_labels, counts = np.unique(labels_np, return_counts=True)

    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(unique_labels, counts, color='skyblue')
    ax1.set_title(f'wykres slupkowy licznosci etykiet {title_suffix}')
    ax1.set_xlabel('emocja')
    ax1.set_ylabel('liczba obrazow')
    ax1.tick_params(axis='x', rotation=45)

    for i, count in enumerate(counts):
        ax1.text(i, count + 50, str(count), ha='center', va='bottom', fontsize=10)

    ax2.pie(counts,
            labels=unique_labels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'black'})
    ax2.set_title(f'wykres kolowy procentowego udzialu etykiet {title_suffix}')
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def preprocess_data(data_list, labels_list, image_size, label_encoder=None):

    if not data_list:
        print("brak danych do przetworzenia.")
        return None, None, None, None, None


    X = np.array(data_list, dtype='float32')


    X = X.reshape(-1, image_size, image_size, 1)


    X /= 255.0


    if label_encoder is None:
        le = LabelEncoder()
        Y_int = le.fit_transform(labels_list)
    else:
        le = label_encoder
        Y_int = le.transform(labels_list)

    emotion_labels = list(le.classes_)


    Y_encoded = to_categorical(Y_int, num_classes=len(emotion_labels))

    return X, Y_int, Y_encoded, emotion_labels, le

def plot_confusion_matrix(model, X_test, Y_test_int, emotion_labels):

    print("\nGenerowanie macierzy pomyłek...")
    Y_pred_probs = model.predict(X_test, verbose=0)
    Y_pred_classes = np.argmax(Y_pred_probs, axis=1)


    cm = confusion_matrix(Y_test_int, Y_pred_classes)


    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=emotion_labels, yticklabels=emotion_labels)

    plt.title('Znormalizowana Macierz Pomyłek')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.show()


def create_cnn_model(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),


        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      Precision(name='precision'),
                      Recall(name='recall'),
                      MeanSquaredError(name='mse'),
                      MeanAbsoluteError(name='mae')
                  ])

    return model

def run_k_fold_cross_validation(X_train_data, Y_train_data_int, Y_train_data_onehot, input_shape, num_classes,
                                n_splits=5, batch_size=64):

    print(f"\nRozpoczęcie Walidacji Krzyżowej (k={n_splits})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    fold_accuracies = []
    fold_losses = []


    for fold, (train_index, val_index) in enumerate(skf.split(X_train_data, Y_train_data_int), 1):
        print(f"\nPrzebieg (Fold) {fold}/{n_splits}")


        model = create_cnn_model(input_shape, num_classes)
        X_fold_train, X_fold_val = X_train_data[train_index], X_train_data[val_index]
        Y_fold_train, Y_fold_val = Y_train_data_onehot[train_index], Y_train_data_onehot[val_index]

        # trening modelu
        history = model.fit(
            X_fold_train, Y_fold_train,
            batch_size=batch_size,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(X_fold_val, Y_fold_val)
        )

        scores = model.evaluate(X_fold_val, Y_fold_val, verbose=0)

        loss = scores[0]
        accuracy = scores[1]
        precision = scores[2]
        recall = scores[3]
        mse = scores[4]
        mae = scores[5]

        print(f"  -> Wynik Walidacji dla Fold {fold}:")
        print(f"    - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"    - Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"    - MSE: {mse:.4f}, MAE: {mae:.4f}")

        fold_accuracies.append(accuracy)
        fold_losses.append(loss)
        fold_results.append(scores)

    fold_results_np = np.array(fold_results)

    mean_accuracy = np.mean(fold_results_np[:, 1])
    std_accuracy = np.std(fold_results_np[:, 1])
    mean_loss = np.mean(fold_results_np[:, 0])
    std_loss = np.std(fold_results_np[:, 0])

    print("\nPodsumowanie Walidacji Krzyżowej")
    print(f"Średnia Dokładność (Accuracy) po {n_splits} przebiegach: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Średnia Strata (Loss) po {n_splits} przebiegach: {mean_loss:.4f} (±{std_loss:.4f})")

    mean_precision = np.mean(fold_results_np[:, 2])
    std_precision = np.std(fold_results_np[:, 2])
    mean_recall = np.mean(fold_results_np[:, 3])
    std_recall = np.std(fold_results_np[:, 3])

    print(f"Średnia Precyzja (Precision): {mean_precision:.4f} (±{std_precision:.4f})")
    print(f"Średnia Czułość (Recall): {mean_recall:.4f} (±{std_recall:.4f})")

    return mean_accuracy, std_accuracy


def visualize_model_predictions(model, X_data, Y_true_int, class_labels, num_samples=9):
    indices = np.random.choice(len(X_data), num_samples, replace=False)

    X_samples = X_data[indices]
    predictions = model.predict(X_samples, verbose=0)

    plt.figure(figsize=(12, 12))
    plt.suptitle("Predykcje modelu na losowych zdjęciach", fontsize=16)

    for i, idx in enumerate(indices):
        img = X_data[idx].reshape(IMAGE_SIZE, IMAGE_SIZE)
        true_label = class_labels[Y_true_int[idx]]


        pred_idx = np.argmax(predictions[i])
        pred_label = class_labels[pred_idx]
        confidence = predictions[i][pred_idx] * 100

        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray')


        title_color = 'green' if pred_label == true_label else 'red'

        plt.title(f"Prawda: {true_label}\nPred: {pred_label} ({confidence:.1f}%)",
                  color=title_color, fontsize=10)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



all_raw_data, all_raw_labels = load_emotion_data(ALL_DATA_DIR, IMAGE_SIZE)

if not all_raw_labels:
    print("\nnie mozna kontynuowac: brak danych")
else:
    #print("\npodsumowanie danych (calkowity zbior)")
    #data_summary(all_raw_labels)
    #visualize_data_balance(all_raw_labels, " (laczny zbior danych)")

    #print("\npre-processing danych")


    X_all, Y_all_int, Y_all_onehot, emotion_labels, label_encoder = preprocess_data(
        all_raw_data,
        all_raw_labels,
        IMAGE_SIZE,
        label_encoder=None
    )

    #print(f"klasy: {emotion_labels}")
    #print(f"liczba klas: {len(emotion_labels)}")

    #print(f"\npodzial danych (trening/test): {100 * (1 - TEST_SPLIT_RATIO):.0f}% / {100 * TEST_SPLIT_RATIO:.0f}%")


    X_train, X_test, Y_train_int, Y_test_int, Y_train_onehot, Y_test_onehot = train_test_split(
        X_all, Y_all_int, Y_all_onehot,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=Y_all_int  
    )

    print(f"rozmiar zbioru treningowego: {X_train.shape[0]}")
    print(f"rozmiar zbioru testowego: {X_test.shape[0]}")

    input_shape = X_train.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)
    if VALIDATION:
        mean_acc, std_acc = run_k_fold_cross_validation(
            X_train, Y_train_int, Y_train_onehot,
            input_shape, num_classes,
            n_splits=5,
            batch_size=BATCH_SIZE
        )

    final_model = create_cnn_model(input_shape, num_classes)

    #print("\nwizualizacja wplywu augmentacji na dane treningowe...")
    # Wyciągamy same warstwy augmentacji z modelu do testu
    # aug_model = create_augmentation_layers()
    # visualize_augmentation_effect(X_train, aug_model, num_samples=3)

    print("\ntrening modelu na pelnym zbiorze treningowym.")
    history = final_model.fit(
        X_train, Y_train_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )

    # ocena modelu na zbiorze testowym
    scores_test = final_model.evaluate(X_test, Y_test_onehot, verbose=0)

    loss_test = scores_test[0]
    accuracy_test = scores_test[1]
    precision_test = scores_test[2]
    recall_test = scores_test[3]
    mse_test = scores_test[4]
    mae_test = scores_test[5]

    print(f"\nwyniki na zbiorze testowym ({100 * TEST_SPLIT_RATIO:.0f}%):")
    print(f"  - loss: {loss_test:.4f}, accuracy: {accuracy_test:.4f}")
    print(f"  - precision: {precision_test:.4f}, recall: {recall_test:.4f}")
    print(f"  - mse: {mse_test:.4f}, mae: {mae_test:.4f}")

    try:
        final_model.save(MODEL_SAVE_PATH)
        print(f"\nmodel zostal zapisany jako: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"\nblad podczas zapisywania modelu: {e}")

    print("\ntrening zakonczony")

    print("\nGenerowanie wizualizacji predykcji...")
    visualize_model_predictions(
        final_model,
        X_test,
        Y_test_int,
        emotion_labels,
        num_samples=9
    )
    plot_confusion_matrix(final_model, X_test, Y_test_int, emotion_labels)