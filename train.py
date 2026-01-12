import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall, MeanSquaredError, MeanAbsoluteError

IMAGE_SIZE = 48
ALL_DATA_DIR = './images'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
TEST_SPLIT_RATIO = 0.1
MODEL_SAVE_PATH = 'emotion_cnn_model_v2.h5'  # sciezka zapisu modelu

# ustawienie ziarna losowosci
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_emotion_data(data_dir, image_size):
    # wczytuje obrazy z podfolderow
    data_list = []
    labels_list = []

    required_shape = (image_size, image_size)

    discarded_count = 0

    print(f"ladowanie danych z: {data_dir}")

    # przechodzenie przez podfoldery (etykiety emocji)
    for emotion_label in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion_label)

        if os.path.isdir(emotion_path):
            for image_filename in os.listdir(emotion_path):
                if image_filename.lower().endswith(SUPPORTED_FORMATS):
                    image_path = os.path.join(emotion_path, image_filename)

                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                    # weryfikacja poprawnosci wczytanego obrazu
                    if image is None or image.ndim != 2 or image.shape != required_shape:
                        discarded_count += 1
                        continue

                    data_list.append(image)
                    labels_list.append(emotion_label)

    print(f"\nzaladowano {len(data_list)} obrazow.")
    print(f"liczba pominietych plikow: {discarded_count}")

    return data_list, labels_list


def data_summary(labels):
    # wyswietla podsumowanie klas
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
    # tworzy wykresy balansowania danych
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
    # przetwarza dane: normalizacja, kodowanie etykiet (label encoder + one-hot)
    if not data_list:
        print("brak danych do przetworzenia.")
        return None, None, None, None, None

    # konwersja i normalizacja
    X = np.array(data_list, dtype='float32')
    X = X.reshape(-1, image_size, image_size, 1)
    X /= 255.0

    # kodowanie etykiet (label encoding)
    if label_encoder is None:
        le = LabelEncoder()
        Y_int = le.fit_transform(labels_list)
    else:
        le = label_encoder
        Y_int = le.transform(labels_list)

    emotion_labels = list(le.classes_)

    # kodowanie one-hot
    Y_encoded = to_categorical(Y_int, num_classes=len(emotion_labels))

    return X, Y_int, Y_encoded, emotion_labels, le


# tworzenie modelu konwolucyjnej sieci neuronowej
def create_cnn_model(input_shape, num_classes):
    # sekwencyjny model cnn
    model = Sequential([
        # warstwy konwolucyjne
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

        # warstwy geste (dense)
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # warstwa wyjsciowa
        Dense(num_classes, activation='softmax')
    ])

    # kompilacja modelu
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


# glowna logika
all_raw_data, all_raw_labels = load_emotion_data(ALL_DATA_DIR, IMAGE_SIZE)

if not all_raw_labels:
    print("\nnie mozna kontynuowac: brak danych")
else:
    print("\npodsumowanie danych (calkowity zbior)")
    data_summary(all_raw_labels)
    visualize_data_balance(all_raw_labels, " (laczny zbior danych)")

    print("\npre-processing danych")

    # przetwarzanie calego zbioru (normalizacja, kodowanie etykiet)
    X_all, Y_all_int, Y_all_onehot, emotion_labels, label_encoder = preprocess_data(
        all_raw_data,
        all_raw_labels,
        IMAGE_SIZE,
        label_encoder=None
    )

    print(f"klasy: {emotion_labels}")
    print(f"liczba klas: {len(emotion_labels)}")

    print(f"\npodzial danych (trening/test): {100 * (1 - TEST_SPLIT_RATIO):.0f}% / {100 * TEST_SPLIT_RATIO:.0f}%")

    # podzial na zbior treningowy i testowy
    X_train, X_test, Y_train_int, Y_test_int, Y_train_onehot, Y_test_onehot = train_test_split(
        X_all, Y_all_int, Y_all_onehot,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=Y_all_int  # utrzymanie proporcji klas
    )

    print(f"rozmiar zbioru treningowego: {X_train.shape[0]}")
    print(f"rozmiar zbioru testowego: {X_test.shape[0]}")

    input_shape = X_train.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)

    # tworzenie i trening modelu na zbiorze treningowym
    final_model = create_cnn_model(input_shape, num_classes)

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

    # zapisanie modelu
    try:
        final_model.save(MODEL_SAVE_PATH)
        print(f"\nmodel zostal zapisany jako: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"\nblad podczas zapisywania modelu: {e}")

    print("\ntrening zakonczony")