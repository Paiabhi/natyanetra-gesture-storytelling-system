import cv2
import mediapipe as mp
import serial

from gesture_classifier_ml import GestureClassifier

classifier = GestureClassifier(
    "mudra_model.pt"
)

try:
    ser = serial.Serial(
        "COM5",
        9600
    )
except:
    ser = None

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        landmarks = []

        for lm in hand.landmark:

            landmarks.extend(
                [lm.x,lm.y,lm.z]
            )

        gesture = classifier.predict(
            landmarks
        )

        cv2.putText(
            frame,
            gesture,
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        if ser:

            ser.write(
                gesture.encode()
            )

    cv2.imshow(
        "NatyaNetra",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()
