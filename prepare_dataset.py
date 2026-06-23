import os
import cv2
import mediapipe as mp
import numpy as np
import csv

DATASET_DIR = "dataset"
OUTPUT_FILE = "landmarks.csv"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

rows = []

for mudra in os.listdir(DATASET_DIR):

    mudra_path = os.path.join(
        DATASET_DIR,
        mudra
    )

    for img_name in os.listdir(mudra_path):

        img_path = os.path.join(
            mudra_path,
            img_name
        )

        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand.landmark:

                landmarks.extend(
                    [lm.x, lm.y, lm.z]
                )

            landmarks.append(mudra)

            rows.append(landmarks)

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerows(rows)

print("Dataset Created")
