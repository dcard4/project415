import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize hand detection
with mp_hands.Hands(
    max_num_hands=1,  # Detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    print("Press 'space' to take a picture and end the program.")

    captured_frame = None  # Variable to store the captured frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        results = hands.process(rgb_frame)

        # Draw hand landmarks if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Optional: Highlight a bounding box around the hand
                h, w, c = frame.shape
                hand_x = [landmark.x * w for landmark in hand_landmarks.landmark]
                hand_y = [landmark.y * h for landmark in hand_landmarks.landmark]
                x_min, x_max = int(min(hand_x)), int(max(hand_x))
                y_min, y_max = int(min(hand_y)), int(max(hand_y))

                # Draw a rectangle around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Hand Detection', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar pressed
            captured_frame = frame  # Save the current frame
            print("Picture captured. Exiting...")
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

    # Show the captured hand in a separate window if a frame was captured
    if captured_frame is not None:
        cv2.imshow('Captured Hand', captured_frame)
        cv2.waitKey(0)  # Wait for any key press to close
        cv2.destroyAllWindows()
