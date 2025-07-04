import argparse
import torch
from f5_tts.api import  F5_TTS, F5_Vocoder
from F5.text import text_to_sequence
import numpy as np
from scipy.io.wavfile import write as write_wav

# Этот скрипт будет запускаться в окружении 'f5tts-env'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Текст для синтеза.")
    parser.add_argument("--output_path", type=str, required=True, help="Путь для сохранения .wav файла.")
    args = parser.parse_args()

    print("Загрузка моделей F5TTS...")
    device = "cpu"

    # Загружаем основную модель F5
    model = F5_TTS('F5/logs/rus/', 'rus_ckpt.pth.tar').to(device).eval()
    
    # Загружаем вокодер (он отвечает за преобразование спектрограммы в звук)
    vocoder = F5_Vocoder('F5/logs/vocoder/').to(device).eval()
    print("Модели F5TTS загружены.")

    # Подготовка текста
    text_sequence = torch.LongTensor(text_to_sequence(args.text, ['ru_filters'])).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([text_sequence.size(1)]).to(device)

    print("Синтез речи с помощью F5TTS...")
    with torch.no_grad():
        mel, mel_lengths, _ = model.infer(text_sequence, text_lengths)
        audio = vocoder(mel).squeeze().cpu().numpy()

    # Сохраняем результат
    write_wav(args.output_path, 22050, audio)
    print(f"Файл успешно сгенерирован: {args.output_path}")
