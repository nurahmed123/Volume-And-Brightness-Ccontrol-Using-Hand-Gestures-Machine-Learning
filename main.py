import cv2
import numpy as np
import mediapipe as mp
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
rcd = 100
BrightnessDown = '''
    tell application "System Events"
        key code 145
    end tell
'''
BrightnessUp = '''
    tell application "System Events"
        key code 144
    end tell
'''

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to set system volume and brightness
def set_system_volume(volume):
    subprocess.run(['osascript', '-e', f'set volume output volume {volume}'])
    
def set_system_brightness(brightness):
    global rcd
    if(brightness > rcd):
        subprocess.run(['osascript', '-e', BrightnessUp], check=True)
    elif(brightness == rcd):return
    else:
        subprocess.run(['osascript', '-e', BrightnessDown], check=True)
    rcd = brightness

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip horizontally

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            h, w, _ = img.shape

            # Get coordinates of thumb tip and index finger tip
            thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            middle_trip = (int(landmarks[12].x * w), int(landmarks[12].y * h))

            # Draw circles on thumb tip and index finger tip
            cv2.circle(img, thumb_tip, 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, index_tip, 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, middle_trip, 10, (255, 255, 255), cv2.FILLED)
            
            #TODO: testing
            print(thumb_tip)
            print(index_tip)
            print(middle_trip)

            # Calculate the distance between thumb tip and index finger tip
            distance = calculate_distance(thumb_tip, index_tip)
            brightnessDistance = calculate_distance(thumb_tip, middle_trip)

            # Volume control logic (distance range can be adjusted)
            vol = np.interp(distance, [20, 200], [0, 100])
            brightness = np.interp(brightnessDistance, [20, 200], [0, 100])
            set_system_volume(int(vol))
            set_system_brightness(int(brightness))

            # Display volume level
            cv2.putText(img, f'Volume: {int(vol)}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 155, 255), 2)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
