import cv2
from utils import HandProcessor
import joblib
from pathlib import Path

class HandSignRecognizer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
        self.model = joblib.load(self.model_path)
        self.hand_processor = HandProcessor()

    def process_video(self, source=0):  # 0 para webcam
        cap = cv2.VideoCapture(source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, processed_frame = self.hand_processor.process_frame(frame)
            
            if landmarks:
                prediction = self.model.predict([landmarks])[0]
                cv2.putText(
                    processed_frame,
                    f"Letra: {prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Reconhecimento de Libras", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 