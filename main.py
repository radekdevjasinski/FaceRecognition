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

IMAGE_SIZE = 48
TRAIN_DATA_DIR = './images/train'
VALIDATION_DATA_DIR = './images/validation'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
SAMPLES_TO_GENERATE_PER_CLASS = 50
RANDOM_SEED = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_emotion_data(data_dir, image_size):
    data_list = []
    labels_list = []

    required_shape = (image_size, image_size)

    discarded_count = 0
    total_files = 0

    print(f"Ładowanie danych z: {data_dir}")

    for emotion_label in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion_label)

        if os.path.isdir(emotion_path):
            for image_filename in os.listdir(emotion_path):
                if image_filename.lower().endswith(SUPPORTED_FORMATS):
                    total_files += 1
                    image_path = os.path.join(emotion_path, image_filename)

                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                    if image is None:
                        discarded_count += 1
                        continue

                    if image.ndim != 2:
                        discarded_count += 1
                        continue

                    if image.shape != required_shape:
                        discarded_count += 1
                        continue

                    data_list.append(image)
                    labels_list.append(emotion_label)

    print(f"\nZaładowano {len(data_list)} obrazów.")
    print(f"Liczba pominiętych plików: {discarded_count}")

    return data_list, labels_list


def data_summary(labels, dataset_name=""):
    if labels:
        labels_np = np.array(labels)

        print(f"\nPodsumowanie danych {dataset_name}")

        unique_labels, counts = np.unique(labels_np, return_counts=True)
        emotion_counts = dict(zip(unique_labels, counts))

        print(f"\nLiczba etykiet: {labels_np.shape[0]}")

        print("Liczebność dla każdej emocji:")

        for label, count in emotion_counts.items():
            print(f"  - {label:<10}: {count} obrazów")

        print(f"\nCałkowita liczba klas: {len(unique_labels)}")
    else:
        print(f"\nBrak danych do przetworzenia w zestawie {dataset_name}.")


def visualize_data_balance(labels, title_suffix):
    labels_np = np.array(labels)
    unique_labels, counts = np.unique(labels_np, return_counts=True)

    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(unique_labels, counts, color='skyblue')
    ax1.set_title(f'Wykres Słupkowy Liczebności Etykiet {title_suffix}')
    ax1.set_xlabel('Emocja')
    ax1.set_ylabel('Liczba obrazów')
    ax1.tick_params(axis='x', rotation=45)

    for i, count in enumerate(counts):
        ax1.text(i, count + 50, str(count), ha='center', va='bottom', fontsize=10)

    wedges, texts, autotexts = ax2.pie(counts,
                                       labels=unique_labels,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       wedgeprops={'edgecolor': 'black'})
    ax2.set_title(f'Wykres Kołowy Procentowego Udziału Etykiet {title_suffix}')
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def preprocess_data(data_list, labels_list, image_size, label_encoder=None):

    if not data_list:
        print("Brak danych do przetworzenia.")
        return None, None, None, None

    # konwersja i normalizacja
    X = np.array(data_list, dtype='float32')
    X = X.reshape(-1, image_size, image_size, 1)
    X /= 255.0

    # kodowanie etykiet
    if label_encoder is None:
        le = LabelEncoder()
        Y_int = le.fit_transform(labels_list)
        emotion_labels = list(le.classes_)
    else:
        le = label_encoder
        Y_int = le.transform(labels_list)
        emotion_labels = list(le.classes_)

    Y_encoded = to_categorical(Y_int, num_classes=len(emotion_labels))

    return X, Y_encoded, emotion_labels, le

# tworzenie modelu konwolucyjnej sieci neuronowej
def create_cnn_model(input_shape, num_classes):
    model = Sequential([

        #warstwy konwolucyjne

        # warstwa wejścowa
        Input(shape=input_shape),

        # warstwa 1
        Conv2D(32, (3, 3), padding='same', activation='relu'), #wykrywanie wzorców
        BatchNormalization(), #normalizacja w celu przyśpieszenia
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)), #zmniejszenie liczby parametrów
        Dropout(0.25), #zapobieganie przeuczenia

        # Warstwa 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Warstwa 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # warstwa klasyfikująca
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # warstwa wyjściowa
        Dense(num_classes, activation='softmax')
    ])

    # kompilacja modelu
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nStruktura Modelu CNN:")
    model.summary()

    return model

# ładowanie Danych

raw_train_data, raw_train_labels = load_emotion_data(TRAIN_DATA_DIR, IMAGE_SIZE)
raw_val_data, raw_val_labels = load_emotion_data(VALIDATION_DATA_DIR, IMAGE_SIZE)

data_summary(raw_train_labels, "TRENINGOWY")
data_summary(raw_val_labels, "WALIDACYJNY")

if not raw_train_labels or not raw_val_labels:
    print("\nNie można kontynuować: brak danych")
else:
    visualize_data_balance(raw_train_labels, " (Treningowy)")
    visualize_data_balance(raw_val_labels, " (Walidacyjny)")


    print("\nPre-processing danych")

    X_train, Y_train, emotion_labels, label_encoder = preprocess_data(
        raw_train_data,
        raw_train_labels,
        IMAGE_SIZE,
        label_encoder=None
    )

    print(f"Kształt danych treningowych: {X_train.shape}")
    print(f"Kształt etykiet treningowych: {Y_train.shape}")
    print(f"Klasy: {emotion_labels}")


    X_val, Y_val, _, _ = preprocess_data(
        raw_val_data,
        raw_val_labels,
        IMAGE_SIZE,
        label_encoder=label_encoder
    )

    print(f"Kształt danych walidacyjnych (X_val): {X_val.shape}")
    print(f"Kształt etykiet walidacyjnych (Y_val): {Y_val.shape}")

    ## tworzenie modelu
    print("\nTworzenie Modelu CNN")
    input_shape = X_train.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)

    model = create_cnn_model(input_shape, num_classes)

    print("\nKonfiguracja modelu zakończona pomyślnie - model jest gotowy do treningu.")

'''
    print("\n--- Rozpoczęcie Treningu (Przykładowa konfiguracja) ---")
    history = model.fit(
        X_train, Y_train,
        batch_size=64,
        epochs=3,
        validation_data=(X_val, Y_val)
    )
'''