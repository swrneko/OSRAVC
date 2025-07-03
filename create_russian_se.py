import torch
import os
import warnings
from TTS.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

warnings.filterwarnings("ignore")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
except ImportError as e:
    pass

print("Создание эталонного русского тембра для Coqui TTS...")

device = "cpu"
output_folder = "checkpoints/base_speakers/RU_XTTS"
os.makedirs(output_folder, exist_ok=True)
os.makedirs('outputs/temp_colors', exist_ok=True)

print("Загрузка Tone Color Converter...")
tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

print("Загрузка модели Coqui TTS...")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# --- Генерируем фразу дефолтным голосом модели ---
# Мы не указываем speaker_wav, и XTTS использует свой усредненный голос
reference_text = '''


Душа моя озарена неземной радостью, как эти чудесные весенние утра, которыми я наслаждаюсь от всего сердца. Я совсем один и блаженствую в здешнем краю, словно созданном для таких, как я. Я так счастлив, мой друг, так упоен ощущением покоя, что искусство мое страдает от этого. Ни одного штриха не мог бы я сделать
'''
reference_wav_path = os.path.join(output_folder, "reference_xtts.wav")

print(f"Генерация эталонной фразы в файл: {reference_wav_path}")
tts_model.tts_to_file(text=reference_text,
                      file_path=reference_wav_path,
                      speaker_wav='/home/swrneko/Documents/30alex_egorov.wav',
                      language="ru")

# --- Извлекаем тембр и сохраняем ---
output_se_path = os.path.join(output_folder, "ru_xtts_se.pth")
print(f"Извлечение тембра и сохранение в: {output_se_path}")

source_se, _ = se_extractor.get_se(reference_wav_path, tone_color_converter, target_dir='outputs/temp_colors', vad=True)
torch.save(source_se, output_se_path)

print(f"\nУСПЕХ! Создан файл эталонного русского тембра: {output_se_path}\n")
