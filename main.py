import cv2
import mediapipe as mp
import numpy as np
import os
import random
import time

#normalize landmarks, turn them into pixel locations
def normalize_landmarks(landmarks, img_width, img_height):
    landmarks = np.array([[lm.x * img_width, lm.y * img_height] for lm in landmarks])

    #we want the origin to be at the wrist
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist

    #max distance from any landmark
    max_value = np.max(np.linalg.norm(normalized_landmarks, axis=1))

    #normalized landmarks
    return normalized_landmarks / max_value


def extract_landmarks_from_images(folder_path):
    #landmarks for each image
    landmarks_list = []

    #the names of each image png
    image_names = []

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        for image_name in os.listdir(folder_path):

            #add the image to the list
            image_names.append(image_name)

            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)

            #make sure theres an image
            if image is None:
                continue

            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                #normalize the landmarks for the image
                for hand_landmarks in results.multi_hand_landmarks:
                    normalized = normalize_landmarks(hand_landmarks.landmark, w, h)

                    #add the landmarks to the list
                    landmarks_list.append(normalized)

    #return the landmarks for the image along with image names
    return landmarks_list, image_names


def calculate_similarity(landmarks1, landmarks2):
    #get the similarity of the landmarks through finding their distance
    return np.linalg.norm(landmarks1 - landmarks2, axis=1).mean()


def play_game(reference_landmarks, reference_names):


    #get a random letter to play the game with
    i = random.randint(0, 25)
    current_letter = reference_names[i]
    current_letter = current_letter[:-4]

    score = 0

    cap = cv2.VideoCapture(0)

    # set a timer
    start_time = time.time()

    #only allow one hand at a time
    #the alphabet only uses one hand
    with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
        #process each frame
        while cap.isOpened():

            #keep the timer running for only 30 seconds
            if time.time() - start_time > 30:
                break

            ret, frame = cap.read()

            #flip the frame, this allows for a selfie like experience
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # display the current time left
            time_left = 30 - (time.time() - start_time)

            cv2.putText(frame, f"Time left: {time_left}", (200, 425), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)


            #display the current letter to be displayed
            cv2.putText(frame, f"Show: {current_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            #display the score
            cv2.putText(frame, f"Score: {score}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)


            #check if there is a hand present
            if results.multi_hand_landmarks:

                #look at the hands every landmark
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    #normalize the landmarks
                    normalized_detected = normalize_landmarks(hand_landmarks.landmark, w, h)

                    #compare the similarities between the landmarks of this hand and the letters dataset
                    similarities = [calculate_similarity(normalized_detected, ref) for ref in reference_landmarks]

                    #find the most similar letter
                    best_match_index = np.argmin(similarities)
                    best_match_name = reference_names[best_match_index]
                    best_match_name = best_match_name[:-4]

                    cv2.putText(frame, f"Detected: {best_match_name}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)

                    if best_match_name == current_letter:

                        #increment the score and get a new letter
                        score += 1
                        i = random.randint(0, 25)
                        current_letter = reference_names[i]
                        current_letter = current_letter[:-4]
                        break

            cv2.imshow('Hand Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    run_game()


def detect_letters(reference_landmarks, reference_names):
    cap = cv2.VideoCapture(0)

    with mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    normalized_detected = normalize_landmarks(hand_landmarks.landmark, w, h)
                    similarities = [calculate_similarity(normalized_detected, ref) for ref in reference_landmarks]
                    best_match_index = np.argmin(similarities)
                    best_match_name = reference_names[best_match_index]
                    best_match_name = best_match_name[:-4]

                    cv2.putText(frame, f"Detected: {best_match_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Letter Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    run_game()


def game_choice(choice):
    if choice == '1':
        play_game(reference_landmarks, reference_names)
    elif choice == '2':
        detect_letters(reference_landmarks, reference_names)

def prompt_user():
    print("Welcome!")
    print("Press 1 to play the sign language game\nPress 2 to practice your sign language skills.")
    choice = input()
    return choice

def run_game():
    choice = prompt_user()
    game_choice(choice)


if __name__ == "__main__":
    reference_folder = 'letters'
    reference_landmarks, reference_names = extract_landmarks_from_images(reference_folder)

    run_game()


