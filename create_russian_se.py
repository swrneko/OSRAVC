import torch
import silero
import os
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Этот скрипт нужно запустить всего один раз!

print("Создание эталонного русского тембра...")

device = "cpu"

# Создаем папки, если их нет
output_folder = "checkpoints/base_speakers/RU"
os.makedirs(output_folder, exist_ok=True)

# --- Загружаем необходимые модели ---
tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

language = 'ru'
model_id = 'v4_ru'
speaker = 'eugene'
sample_rate = 48000
silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                 model='silero_tts',
                                 language=language,
                                 speaker=model_id)
silero_model.to(device)

# --- Генерируем длинную, нейтральную фразу для анализа ---
reference_text = '''
В жизни каждого человека наступает момент, когда ему нужно сделать важный выбор. Это может быть выбор профессии, места для проживания, круга общения или даже решение о том, как провести свободное время. И хотя мы часто не осознаём важности каждого выбора, каждый из них влияет на наше будущее. Особенно это актуально, когда перед нами стоит решение, касающееся долгосрочных планов, изменений в жизни или переезда в другое место. Многие из этих решений могут повлиять не только на нас, но и на людей вокруг, на нашу семью, друзей и коллег.
'''
reference_wav_path = os.path.join(output_folder, "reference.wav")
silero_model.save_wav(text=reference_text,
                      speaker=speaker,
                      sample_rate=sample_rate,
                      audio_path=reference_wav_path)

# --- Извлекаем тембр и сохраняем его в файл .pth ---
output_se_path = os.path.join(output_folder, "ru_default_se.pth")

# Запускаем извлечение
source_se, _ = se_extractor.get_se(reference_wav_path, tone_color_converter, target_dir='outputs/temp_colors', vad=True)

# Сохраняем тензор
torch.save(source_se, output_se_path)

print(f"Успешно создан файл эталонного русского тембра: {output_se_path}")
print("Теперь можно удалить этот скрипт (create_russian_se.py) и файл reference.wav.")
