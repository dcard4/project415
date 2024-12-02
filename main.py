import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks, img_width, img_height):
    # Convert landmarks to a numpy array and normalize relative to wrist (landmark 0)
    landmarks = np.array([[lm.x * img_width, lm.y * img_height] for lm in landmarks])
    wrist = landmarks[0]  # Assume wrist is the first point
    normalized_landmarks = landmarks - wrist  # Normalize to wrist
    max_value = np.max(np.linalg.norm(normalized_landmarks, axis=1))  # Scale normalization
    return normalized_landmarks / max_value


def extract_landmarks_from_images(folder_path):
    landmarks_list = []  # Store landmarks for each image
    image_names = []  # Store the corresponding image names
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    normalized = normalize_landmarks(hand_landmarks.landmark, w, h)
                    landmarks_list.append(normalized)
                    image_names.append(image_name)
    return landmarks_list, image_names


def calculate_similarity(landmarks1, landmarks2):
    # Calculate Euclidean distance between two sets of landmarks
    return np.linalg.norm(landmarks1 - landmarks2, axis=1).mean()


# Preprocess reference images
reference_folder = 'letters'
reference_landmarks, reference_names = extract_landmarks_from_images(reference_folder)

# Open the camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    print("Press 'space' to take a picture and end the program.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        results = hands.process(rgb_frame)

        # Draw hand landmarks if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Normalize detected hand landmarks
                normalized_detected = normalize_landmarks(hand_landmarks.landmark, w, h)

                # Compare with reference landmarks
                similarities = [calculate_similarity(normalized_detected, ref) for ref in reference_landmarks]
                best_match_index = np.argmin(similarities)
                best_match_name = reference_names[best_match_index]

                # Display the best match name on the frame
                cv2.putText(frame, f"Best match: {best_match_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the result
        cv2.imshow('Hand Detection', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar pressed
            print("Exiting...")
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
