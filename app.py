import gradio as gr
import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
import warnings

# --- Подавляем предупреждения ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Настройка безопасности PyTorch ---
# Добавляем ВСЕ классы XTTS в "белый список" для загрузчика моделей.
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig 
    from TTS.tts.models.xtts import XttsArgs
    
    # Добавляем все три класса в список доверенных:
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]) # <-- ДОБАВЛЕН ТРЕТИЙ КЛАСС
except ImportError as e:
    print(f"Не удалось импортировать классы TTS: {e}. Возможно, используется старая версия TTS.")
    pass

# --- Импорты ---
from TTS.api import TTS
from ruaccent import RUAccent
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# --- 1. Настройка и загрузка моделей ---
print("Загрузка моделей...")
device = "cpu"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

accent_model = RUAccent()
accent_model.load(omograph_model_size='big_poetry', use_dictionary=True)

print("Загрузка модели Coqui TTS (может занять время)...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
print("Модель Coqui TTS успешно загружена.")

print("Все модели успешно загружены.")

# --- 2. Функционал ---
def accent_text(text):
    if not text: return ""
    return accent_model.process_all(text)

def generate_russian_voice(accented_text, reference_audio_data):
    if not accented_text: raise gr.Error("Введите текст.")
    if reference_audio_data is None: raise gr.Error("Загрузите аудио-референс.")

    print(f"Генерация речи: '{accented_text}'")
    
    ref_sample_rate, ref_audio_array = reference_audio_data
    temp_reference_path = os.path.join(output_dir, "temp_reference.wav")
    write_wav(temp_reference_path, ref_sample_rate, ref_audio_array.astype(np.int16))

    base_speech_path = os.path.join(output_dir, 'base_russian_speech.wav')
    tts.tts_to_file(text=accented_text, file_path=base_speech_path, language="ru", speaker_wav=temp_reference_path)

    target_se, _ = se_extractor.get_se(temp_reference_path, tone_color_converter, target_dir=os.path.join(output_dir, 'temp_colors'), vad=True)
    source_se, _ = se_extractor.get_se(base_speech_path, tone_color_converter, target_dir=os.path.join(output_dir, 'temp_colors'), vad=True)

    output_path = os.path.join(output_dir, 'generated_russian_voice.wav')
    tone_color_converter.convert(
        audio_src_path=base_speech_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message="@MyShell")

    print("Голос успешно сгенерирован.")
    return output_path

# --- 3. Интерфейс ---
with gr.Blocks(title="Russian OpenVoice UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OpenVoice: Клонирование голоса на русском языке")
    gr.Markdown("1. Введите текст. 2. Нажмите «Расставить ударения». 3. Исправьте ударения. 4. Загрузите аудио-референс. 5. Нажмите «Генерировать».")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Шаг 1: Текст", placeholder="Мама мыла раму.")
            accent_button = gr.Button("Шаг 2: Ударения", variant="secondary")
            accented_text_output = gr.Textbox(label="Шаг 3: Проверка ударений", placeholder="М+ама м+ыла р+аму.", interactive=True)
            audio_input = gr.Audio(label="Шаг 4: Аудио-референс", type="numpy")
            generate_button = gr.Button("Шаг 5: Генерировать", variant="primary")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Результат")

    accent_button.click(fn=accent_text, inputs=[text_input], outputs=[accented_text_output])
    generate_button.click(fn=generate_russian_voice, inputs=[accented_text_output, audio_input], outputs=[audio_output], api_name="russian_voice_clone")

# --- 4. Запуск ---
if __name__ == "__main__":
    demo.launch(share=True)
