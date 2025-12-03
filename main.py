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
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.metrics import Precision, Recall, MeanSquaredError, MeanAbsoluteError

IMAGE_SIZE = 48
ALL_DATA_DIR = './images'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
SAMPLES_TO_GENERATE_PER_CLASS = 50
RANDOM_SEED = 42
K_FOLDS = 5
EPOCHS = 3
TEST_SPLIT_RATIO = 0.1

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

    # przechodzenie przez podfoldery
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


def data_summary(labels):
    if labels:
        labels_np = np.array(labels)

        print(f"\nPodsumowanie danych")

        unique_labels, counts = np.unique(labels_np, return_counts=True)
        emotion_counts = dict(zip(unique_labels, counts))

        print(f"\nLiczba etykiet: {labels_np.shape[0]}")

        print("Liczebność dla każdej emocji:")

        for label, count in emotion_counts.items():
            print(f"  - {label:<10}: {count} obrazów")

        print(f"\nCałkowita liczba klas: {len(unique_labels)}")
    else:
        print(f"\nBrak danych do przetworzenia.")


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
        return None, None, None, None, None

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

    # kodowanie One-Hot
    Y_encoded = to_categorical(Y_int, num_classes=len(emotion_labels))

    return X, Y_int, Y_encoded, emotion_labels, le


# tworzenie modelu konwolucyjnej sieci neuronowej
def create_cnn_model(input_shape, num_classes):
    # model sekwencyjny, warstwy są ułożone jedna po drugiej
    model = Sequential([

        # warstwy konwolucyjne
        # definiuje kształt danych wejściowych (48, 48, 1).
        Input(shape=input_shape),

        # warstwa konwolucyjna (wykrywanie cech)
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        # normalizuje aktywacje wyjściowe, przyspieszając trening i stabilizując model
        BatchNormalization(),


        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        # warstwa zmniejszająca wymiary (downsampling), redukuje liczbę parametrów i cech
        MaxPooling2D(pool_size=(2, 2)),
        # ustawia losowy ułamek (25%) neuronów na 0 w każdej iteracji, zapobiegając przeuczeniu
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

        # warstwa ukryta, ucząca się nieliniowych kombinacji cech
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # warstwa wyjściowa
        # funkcja aktywacji zamieniająca wyjścia na rozkład prawdopodobieństw (suma = 1.0).
        Dense(num_classes, activation='softmax')
    ])

    # kompilacja modelu
    model.compile(optimizer='adam',
                  # Loss: Funkcja do minimalizacji, mierząca "odległość" między przewidywanymi a prawdziwymi rozkładami prawdopodobieństwa.
                  loss='categorical_crossentropy',
                  metrics=[
                      # Accuracy (Dokładność): Proporcja poprawnie sklasyfikowanych obrazów do wszystkich obrazów.
                      'accuracy',

                      # Precision (Precyzja): Jak duża część obrazów sklasyfikowanych jako pozytywne faktycznie nimi były.
                      Precision(name='precision'),

                      # Recall (Czułość): Jak duża część faktycznie pozytywnych obrazów została poprawnie zidentyfikowana.
                      Recall(name='recall'),

                      # Mean Squared Error (MSE): Średnia kwadratów różnic między przewidywaniami modelu a rzeczywistymi wartościami (ocena "pewności" modelu).
                      MeanSquaredError(name='mse'),

                      # Mean Absolute Error (MAE): Średnia bezwzględnych różnic między przewidywaniami a rzeczywistymi wartościami (mniej wrażliwa na duże błędy niż MSE).
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

        # inicjalizacja nowego modelu dla każdego podziału
        model = create_cnn_model(input_shape, num_classes)

        # podział danych na zbiory treningowe i walidacyjne dla danego 'fold'
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

        # ocena modelu na zbiorze walidacyjnym z danego 'fold'
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

    # obliczenie średniej i odchylenia standardowego
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




# GŁÓWNA LOGIKA
all_raw_data, all_raw_labels = load_emotion_data(ALL_DATA_DIR, IMAGE_SIZE)

if not all_raw_labels:
    print("\nNie można kontynuować: brak danych")
else:
    data_summary(all_raw_labels)
    visualize_data_balance(all_raw_labels, " (Łączny Zbiór Danych)")

    print("\nPre-processing danych")

    # przetwarzanie całego zbioru (normalizacja, kodowanie etykiet)
    X_all, Y_all_int, Y_all_onehot, emotion_labels, label_encoder = preprocess_data(
        all_raw_data,
        all_raw_labels,
        IMAGE_SIZE,
        label_encoder=None
    )

    print(f"Klasy: {emotion_labels}")
    print(f"Liczba klas: {len(emotion_labels)}")

    print(f"\nPodział Danych (Trening/Test): 90% / 10%")

    X_train, X_test, Y_train_int, Y_test_int, Y_train_onehot, Y_test_onehot = train_test_split(
        X_all, Y_all_int, Y_all_onehot,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=Y_all_int  # Utrzymanie proporcji klas
    )

    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} ({100 * (1 - TEST_SPLIT_RATIO):.0f}%)")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} ({100 * TEST_SPLIT_RATIO:.0f}%)")

    # walidacja Krzyżowa na Zbiorze Treningowym
    input_shape = X_train.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)

    # uruchomienie Walidacji Krzyżowej
    mean_acc, std_acc = run_k_fold_cross_validation(
        X_train, Y_train_int, Y_train_onehot,
        input_shape,
        num_classes,
        n_splits=K_FOLDS,
    )

    print("\nOcena na Zbiorze Testowym")

    # wytrenowanie ostatecznego modelu na całym zbiorze treningowym
    final_model = create_cnn_model(input_shape, num_classes)

    print("\nTrening modelu na pełnym zbiorze treningowym.")
    history = final_model.fit(
        X_train, Y_train_onehot,
        batch_size=64,
        epochs=EPOCHS,
        verbose=1
    )

    scores_test = final_model.evaluate(X_test, Y_test_onehot, verbose=0)

    loss_test = scores_test[0]
    accuracy_test = scores_test[1]
    precision_test = scores_test[2]
    recall_test = scores_test[3]
    mse_test = scores_test[4]
    mae_test = scores_test[5]

    print(f"\nWyniki na Zbiorze Testowym (10%):")
    print(f"  - Loss: {loss_test:.4f}, Accuracy: {accuracy_test:.4f}")
    print(f"  - Precision: {precision_test:.4f}, Recall: {recall_test:.4f}")
    print(f"  - MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")

    print("\nTrening zakończony")