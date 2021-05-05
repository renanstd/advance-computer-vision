import cv2
import mediapipe
import time


WEBCAM_URL = "http://192.168.1.32:4747/video"


class HandDetector():
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.draw = mediapipe.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return image

    def find_position(self, image, hand_number=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, landmark in enumerate(my_hand.landmark):
                # Pegar as medidas da imagem
                image_height, image_width, image_channels = image.shape
                landmark_x, landmark_y = int(landmark.x * image_width), int(landmark.y * image_height)
                landmark_list.append([id, landmark_x, landmark_y])
                if draw:
                    cv2.circle(
                        image,
                        (landmark_x, landmark_y),
                        15,
                        (255, 0, 255),
                        cv2.FILLED
                    )
        return landmark_list


def main():
    previous_time = 0
    current_time = 0

    capture = cv2.VideoCapture(WEBCAM_URL)
    detector = HandDetector()

    while True:
        _, image = capture.read()

        image = detector.find_hands(image)
        positions = detector.find_position(image)
        print(positions)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            image,                      # Image
            str(int(fps)),              # Text
            (10, 70),                   # Position
            cv2.FONT_HERSHEY_COMPLEX,   # Font
            1,                          # Scale
            (255, 0, 255),              # Color
            2                           # Thickness
        )

        cv2.imshow("Image", image)
        res = cv2.waitKey(1)
        if res & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
