import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Tuple

class HandProcessor:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Processa apenas uma mão
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame: np.ndarray) -> Tuple[List[float], np.ndarray]:
        # Redimensiona para tamanho consistente
        frame = cv2.resize(frame, (640, 480))
        
        # Converte para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Pega primeira mão
            
            # Normaliza em relação ao ponto central da mão
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            for lm in hand_landmarks.landmark:
                # Normaliza coordenadas em relação ao ponto base
                landmarks.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])
                
            # Desenha landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        return landmarks, frame 