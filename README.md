# Projeto de Reconhecimento de Libras

Este projeto implementa um sistema de reconhecimento de gestos da Língua Brasileira de Sinais (Libras) utilizando visão computacional e aprendizado de máquina. O sistema é capaz de capturar gestos em tempo real através de uma webcam e reconhecer as letras do alfabeto em Libras.


# Feito por 

João Antonio Rihan rm99656 

Rodrigo Fernandes rm550816 

Eric Rodrigues rm550249 

Victória Pizza rm550609

## Estrutura do Projeto

```
.
├── programas/           # Código fonte do projeto
│   ├── main.py         # Script principal de execução
│   ├── collect_data.py # Script para coleta de dados de treinamento
│   ├── train_model.py  # Script para treinamento do modelo
│   ├── recognize.py    # Script para reconhecimento de gestos
│   └── utils.py        # Funções utilitárias
├── dados/              # Diretório para armazenamento de dados
├── modelo_letras.pkl   # Modelo treinado para reconhecimento
└── video.mp4           # Vídeo de exemplo
```

## Requisitos

- Python 3.x
- OpenCV (cv2)
- MediaPipe
- NumPy
- Scikit-learn
- Pickle

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/joaorihan/Libras-CP2.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Coleta de Dados
Para coletar dados de treinamento:
```bash
python programas/collect_data.py
```

### Treinamento do Modelo
Para treinar o modelo com os dados coletados:
```bash
python programas/train_model.py
```

### Reconhecimento em Tempo Real
Para executar o reconhecimento de gestos em tempo real:
```bash
python programas/main.py
```

## Funcionalidades

- Captura de gestos em tempo real através da webcam
- Processamento de imagens para extração de características
- Reconhecimento de letras do alfabeto em Libras
- Interface visual para feedback do usuário

## Arquivos Principais

- `main.py`: Script principal que integra todas as funcionalidades
- `collect_data.py`: Implementa a coleta de dados para treinamento
- `train_model.py`: Responsável pelo treinamento do modelo de machine learning
- `recognize.py`: Contém a lógica de reconhecimento de gestos
- `utils.py`: Funções auxiliares para processamento de imagens e dados
