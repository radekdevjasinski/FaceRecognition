import os
import cv2
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

IMAGE_SIZE = 48
TEST_DATA_DIR = './images/train'
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
SAMPLES_TO_GENERATE_PER_CLASS = 50

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

def data_summary(labels):
    if labels:
        labels_np = np.array(labels)

        print(f"\nPodsumowanie danych")

        unique_labels, counts = np.unique(labels_np, return_counts=True)
        emotion_counts = dict(zip(unique_labels, counts))

        print(f"\nLiczba etykiet: {labels_np.shape[0]}")

        print("\nLiczebność dla każdej emocji:")

        for label, count in emotion_counts.items():
            print(f"  - {label:<10}: {count} obrazów")

        print(f"\nCałkowita liczba klas: {len(unique_labels)}")
    else:
        print("\nBrak danych do przetworzenia.")

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

def augment_data(data_list, labels_list, samples_per_label, img_size):

    augmented_data = []
    augmented_labels = []

    augmentation_pairs_map = defaultdict(list)

    grouped_images = defaultdict(list)
    for img, label in zip(data_list, labels_list):
        grouped_images[label].append(img)

    print(f"\nAugmentacja danych (Generowanie {samples_per_label} próbek na klasę)...")

    for label, images in grouped_images.items():
        if not images:
            continue

        samples = random.choices(images, k=samples_per_label)

        for original_img in samples:
            img = original_img.copy()

            # Obrót
            angle = random.uniform(-25, 25)
            M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img_size, img_size),
                                 borderMode=cv2.BORDER_REPLICATE)

            # Skalowanie
            scale = random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), 0, scale)
            img = cv2.warpAffine(img, M, (img_size, img_size),
                                 borderMode=cv2.BORDER_REPLICATE)

            # Szum Gaussa
            row, col = img.shape
            mean = 0
            std_dev = random.uniform(5, 15)

            gauss = np.random.normal(mean, std_dev, (row, col))
            gauss = gauss.reshape(row, col)

            noisy_img = img + gauss
            noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')

            augmented_data.append(noisy_img)
            augmented_labels.append(label)

            augmentation_pairs_map[label].append((original_img, noisy_img))

    print(f"Zakończono augmentację. Dodano {len(augmented_data)} nowych próbek.")

    data_list.extend(augmented_data)
    labels_list.extend(augmented_labels)

    return data_list, labels_list, augmentation_pairs_map

def display_augmentation_examples(augmentation_pairs_map, num_examples=3):

    all_labels = list(augmentation_pairs_map.keys())
    examples_to_show = min(num_examples, len(all_labels))
    selected_labels = random.sample(all_labels, examples_to_show)

    fig, axes = plt.subplots(examples_to_show, 2, figsize=(8, 4 * examples_to_show))

    if examples_to_show == 1:
        axes = np.array([axes])

    for i, label in enumerate(selected_labels):
        if augmentation_pairs_map.get(label):
            original_img, augmented_img = random.choice(augmentation_pairs_map[label])

            axes[i, 0].imshow(original_img, cmap='gray')
            axes[i, 0].set_title(f"Oryginał: {label}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(augmented_img, cmap='gray')
            axes[i, 1].set_title(f"Augmentacja: {label}")
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


raw_data, raw_labels = load_emotion_data(TEST_DATA_DIR, IMAGE_SIZE)

data_summary(raw_labels)
if raw_labels:
    visualize_data_balance(raw_labels, "")

'''
if raw_data:
    raw_data, raw_labels, augmentation_pairs_map = augment_data(
        raw_data,
        raw_labels,
        SAMPLES_TO_GENERATE_PER_CLASS,
        IMAGE_SIZE
    )

    display_augmentation_examples(augmentation_pairs_map, num_examples=3)

data_summary(raw_labels)
if raw_labels:
    visualize_data_balance(raw_labels, "(PO Augmentacji)")
'''