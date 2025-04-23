from collect_data import DataCollector
from train_model import ModelTrainer
from recognize import HandSignRecognizer
import os

def main():
    # Configuração de caminhos
    video_path = "../video.mp4"  # Ajuste o caminho conforme necessário
    data_dir = "../dados"
    model_path = "../modelo_letras.pkl"

    # Menu de opções
    while True:
        print("\nSistema de Reconhecimento de Libras")
        print("1. Coletar dados de treinamento")
        print("2. Treinar modelo")
        print("3. Reconhecer letras (webcam)")
        print("4. Reconhecer letras (vídeo)")
        print("5. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == "1":
            collector = DataCollector(video_path, data_dir)
            collector.collect()
            
        elif choice == "2":
            trainer = ModelTrainer(data_dir, model_path)
            trainer.train()
            
        elif choice == "3":
            recognizer = HandSignRecognizer(model_path)
            recognizer.process_video(0)  # 0 para webcam
            
        elif choice == "4":
            video_path = input("Digite o caminho do vídeo: ")
            if os.path.exists(video_path):
                recognizer = HandSignRecognizer(model_path)
                recognizer.process_video(video_path)
            else:
                print("Arquivo de vídeo não encontrado!")
                
        elif choice == "5":
            break
            
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main() 