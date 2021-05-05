import cv2
import mediapipe
import time


WEBCAM_URL = "http://192.168.1.32:4747/video"


# Método de captura
capture = cv2.VideoCapture(WEBCAM_URL)

# Inicializa módulo de detecção
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands()
# Inicializa utilitário de desenho
draw = mediapipe.solutions.drawing_utils

# Variáveis para cálculo do FPS
previous_time = 0
current_time = 0

while True:
    # Captura e processa a imagem
    success, image = capture.read()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Identifica as mãos usando o mediapipe
    results = hands.process(image_rgb)
    # Processa as landmarks das mãos encontradas
    if results.multi_hand_landmarks:
        # Loop por todas as mãos encontradas
        for hand_landmarks in results.multi_hand_landmarks:
            # Loop por todos os landmarks detectados
            for id, landmark in enumerate(hand_landmarks.landmark):
                # Pegar as medidas da imagem
                image_height, image_width, image_channels = image.shape
                # Calcular as posições dos pixels dos landmarks
                # Para isso, multiplicamos as posições retornadas na landmark
                # pelo tamanho completo da imagem
                landmark_x, landmark_y = int(landmark.x * image_width), int(landmark.y * image_height)
                # Desenhar um círculo nos landmarks das pontas dos dedos
                if id == 4:
                    cv2.circle(image, (landmark_x, landmark_y), 15, (255, 0, 255), cv2.FILLED)
                if id == 8:
                    cv2.circle(image, (landmark_x, landmark_y), 15, (255, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(image, (landmark_x, landmark_y), 15, (255, 0, 255), cv2.FILLED)
                if id == 16:
                    cv2.circle(image, (landmark_x, landmark_y), 15, (255, 0, 255), cv2.FILLED)
                if id == 20:
                    cv2.circle(
                        image,                      # Image
                        (landmark_x, landmark_y),   # Position
                        15,                         # Radius
                        (255, 0, 255),              # Color
                        cv2.FILLED                  # Filling
                    )
            # Desenha as marcações detectadas, com as respectivas conexões
            draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calcular o FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Exibir o FPS
    cv2.putText(
        image,                      # Image
        str(int(fps)),              # Text
        (10, 70),                   # Position
        cv2.FONT_HERSHEY_COMPLEX,   # Font
        1,                          # Scale
        (255, 0, 255),              # Color
        2                           # Thickness
    )

    # Exibe a imagem, e fecha o viewer ao pressionar 'q'
    cv2.imshow("Image", image)
    res = cv2.waitKey(1)
    if res & 0xFF == ord('q'):
        break
