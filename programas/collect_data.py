import cv2
import os
import csv
from utils import HandProcessor
from pathlib import Path

class DataCollector:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.hand_processor = HandProcessor()
        self.intervals = {
            'A': (5, 8),
            'B': (9, 13),
            'C': (14, 19),
            'D': (20, 23),
            'E': (24, 28),
            'F': (29, 33),
            'G': (34, 38),
            'H': (39, 44),
            'I': (45, 49),
            'J': (50, 54),
            'K': (55, 58),
            'L': (59, 63),
            'M': (64, 69),
            'N': (70, 73),
            'O': (74, 78),
            'P': (79, 84),
            'Q': (85, 89),
            'R': (90, 94),
            'S': (95, 98),
            'T': (99, 104),
            'U': (105, 108),
            'V': (109, 114),
            'W': (115, 118),
            'X': (119, 123),
            'Y': (124, 129),
            'Z': (130, 134)
        }

    def collect(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video não encontrado: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Inicializa arquivos CSV
        self._initialize_csv_files()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            for letter, (start, end) in self.intervals.items():
                if start <= current_time <= end:
                    landmarks, processed_frame = self.hand_processor.process_frame(frame)
                    
                    if landmarks:  # Se detectou landmarks
                        self._save_landmarks(letter, landmarks)
                    
                    # Mostra progresso
                    cv2.imshow('Collecting Data', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

    def _initialize_csv_files(self):
        header = ['label'] + [f"{coord}{i}" for i in range(21) for coord in ['x', 'y', 'z']]
        
        for letter in self.intervals.keys():
            csv_path = self.output_dir / f"coleta_{letter}.csv"
            if not csv_path.exists():
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

    def _save_landmarks(self, letter: str, landmarks: list):
        csv_path = self.output_dir / f"coleta_{letter}.csv"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([letter] + landmarks) 