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

# Генерируем нейтральный сэмпл, чтобы получить голос диктора
neutral_speaker_wav = os.path.join(output_folder, "neutral_speaker_sample.wav")
tts_model.tts_to_file(text="Привет.", file_path=neutral_speaker_wav, language="ru")

# Генерируем эталонную фразу этим голосом
reference_text = "Здравствуйте, это стандартный образец голоса для системы клонирования тембра."
reference_wav_path = os.path.join(output_folder, "reference_xtts.wav")
print(f"Генерация эталонной фразы в файл: {reference_wav_path}")
tts_model.tts_to_file(text=reference_text, file_path=reference_wav_path, language="ru", speaker_wav=neutral_speaker_wav)

# Извлекаем и сохраняем тембр
output_se_path = os.path.join(output_folder, "ru_xtts_se.pth")
print(f"Извлечение тембра и сохранение в: {output_se_path}")
source_se, _ = se_extractor.get_se(reference_wav_path, tone_color_converter, target_dir='outputs/temp_colors', vad=True)
torch.save(source_se, output_se_path)

print(f"\nУСПЕХ! Создан файл эталонного русского тембра: {output_se_path}\n")
