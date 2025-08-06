import cv2
import mediapipe as mp
import pygame
import random
import time
import math
import os

# Initialize sound
pygame.mixer.init()
slice_sound = pygame.mixer.Sound('sounds/slice.wav')

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load assets
fruit_images = []
asset_path = 'assets'
for file in os.listdir(asset_path):
    if file.endswith('.png'):
        img = cv2.imread(os.path.join(asset_path, file), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            fruit_images.append(img)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Fruit class
class Fruit:
    def __init__(self, img, x, y, speed):
        self.img = img
        self.x = x
        self.y = y
        self.speed = speed
        self.alive = True
        self.w = img.shape[1]
        self.h = img.shape[0]

    def move(self):
        self.y += self.speed
        if self.y > 480:
            self.alive = False

    def draw(self, frame):
        if self.alive:
            try:
                x1, y1 = int(self.x), int(self.y)
                x2, y2 = x1 + self.w, y1 + self.h

                fruit_alpha = self.img[:, :, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        fruit_alpha * self.img[:, :, c] +
                        (1 - fruit_alpha) * frame[y1:y2, x1:x2, c]
                    )
            except:
                pass

# Main game function
def run_game():
    score = 0
    fruits = []
    trail = []
    prev_x, prev_y = None, None
    spawn_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        curr_x, curr_y = None, None
        swipe = False

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
                curr_x = int(hand_landmark.landmark[8].x * w)
                curr_y = int(hand_landmark.landmark[8].y * h)
                trail.append((curr_x, curr_y))
                if len(trail) > 15:
                    trail.pop(0)

                if prev_x and prev_y:
                    distance = math.hypot(curr_x - prev_x, curr_y - prev_y)
                    if distance > 40:
                        swipe = True

        # Draw hand trail
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 3)

        # Spawn fruit
        if time.time() - spawn_time > 1.5 and len(fruits) < 5:
            fruit_img = random.choice(fruit_images)
            fx = random.randint(0, w - 100)
            speed = random.randint(5, 8)
            fruits.append(Fruit(fruit_img, fx, 0, speed))
            spawn_time = time.time()

        # Move and draw fruits
        for fruit in fruits:
            fruit.move()
            fruit.draw(frame)

            # Check for slicing
            if swipe and curr_x and curr_y and fruit.alive:
                center_x = fruit.x + fruit.w // 2
                center_y = fruit.y + fruit.h // 2
                if abs(center_x - curr_x) < fruit.w // 2 and abs(center_y - curr_y) < fruit.h // 2:
                    fruit.alive = False
                    score += 1
                    slice_sound.play()

        fruits = [f for f in fruits if f.alive]

        # Display score
        cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow("Gesture Fruit Ninja", frame)

        prev_x, prev_y = curr_x, curr_y

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

run_game()
cap.release()
cv2.destroyAllWindows()