import cv2  # For camera module
import mediapipe as mp  # For hand movement, ML framework
import pyautogui  # For accessing the keyboard and mouse
import time  # For managing time between clicks

# Initializing MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define constants for screen width and height
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Parameters for smoothing and debouncing
SMOOTHING_FACTOR = 0.2
DEBOUNCE_TIME = 0.2  # Time to wait to prevent multiple clicks

# For smoothing
previous_x, previous_y = 0, 0
last_action_time = 0

def smooth_movement(current_x, current_y):
    global previous_x, previous_y
    smoothed_x = int(previous_x + SMOOTHING_FACTOR * (current_x - previous_x))
    smoothed_y = int(previous_y + SMOOTHING_FACTOR * (current_y - previous_y))
    previous_x, previous_y = smoothed_x, smoothed_y
    return smoothed_x, smoothed_y

# Function for finger detection
def are_fingers_up(landmarks):
    fingers = []

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers
    for id in range(8, 21, 4):  # Iterating loop for four fingers with a step of 4
        if landmarks[id].y < landmarks[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def main():
    global last_action_time

    cap = cv2.VideoCapture(0)  # Opens the webcam for capturing video.
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the finger states
                fingers = are_fingers_up(hand_landmarks.landmark)
                
                # Check for "Hang Loose" gesture (thumb and pinky extended)
                if fingers == [1, 0, 0, 0, 1]:  # Thumb and pinky up, others down
                    print("Exiting... 'Hang Loose' gesture detected.")
                    break  # Exit the loop to quit the program

                # Get the position of the index finger tip
                index_finger_tip = hand_landmarks.landmark[8]
                index_x = int(index_finger_tip.x * SCREEN_WIDTH)
                index_y = int(index_finger_tip.y * SCREEN_HEIGHT)

                # Smooth cursor movement
                smoothed_x, smoothed_y = smooth_movement(index_x, index_y)

                # Navigation
                if fingers == [0, 1, 0, 0, 0]:  # For one finger up
                    pyautogui.moveTo(smoothed_x, smoothed_y)

                # Scroll Up
                elif fingers == [0, 1, 1, 0, 0]:  # For two fingers up
                    current_time = time.time()
                    if (current_time - last_action_time) > DEBOUNCE_TIME:
                        pyautogui.scroll(100)
                        last_action_time = current_time

                # Scroll Down (Fist)
                elif fingers == [0, 0, 0, 0, 0]:  # No fingers up (fist)
                    current_time = time.time()
                    if (current_time - last_action_time) > DEBOUNCE_TIME:
                        pyautogui.scroll(-100)
                        last_action_time = current_time

                # Click
                elif fingers == [1, 1, 1, 1, 1]:  # Click when all fingers are extended
                    current_time = time.time()
                    if (current_time - last_action_time) > DEBOUNCE_TIME:
                        pyautogui.click()
                        last_action_time = current_time
                        cv2.putText(frame, 'Clicked!', (smoothed_x - 50, smoothed_y - 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with hand landmarks
        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # To exit press 'q'
            break

    cap.release()  # Close webcam
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
