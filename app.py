import gradio as gr
import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
import warnings
import pyloudnorm as pyln

# --- 0. Настройка и глобальные переменные ---
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
except ImportError as e:
    pass


# --- ИЗМЕНЕНИЕ: Настройка многопоточности PyTorch ---
# Устанавливаем количество потоков равным количеству ядер CPU
# Это должно значительно ускорить инференс на CPU
if torch.get_num_threads() > 1:
    torch.set_num_threads(os.cpu_count())
    print(f"🔥 Установлено {os.cpu_count()} потоков для PyTorch.")

print("Инициализация...")
DEVICE = "cpu"
OUTPUT_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- 1. Загрузка моделей ---
from TTS.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

print("🔄 Загрузка Coqui XTTS...")
MODEL_NAME = "xttsv2_banana"
MODEL_PATH = "xttsv2_banana"
CONFIG_PATH = "xttsv2_banana/config.json"
tts = TTS(model_name=MODEL_NAME, model_path=MODEL_PATH, config_path=CONFIG_PATH).to(DEVICE)


print("🔄 Загрузка OpenVoice Converter...")
tone_color_converter = ToneColorConverter('checkpoints/converter/config.json', device=DEVICE)
tone_color_converter.load_ckpt('checkpoints/converter/checkpoint.pth')

print("✅ Все модели готовы.")

# --- 2. Функционал ---
def generate_voice(text: str, reference_audio: tuple):
    if not text.strip():
        raise gr.Error("Пожалуйста, введите текст.")
    if reference_audio is None:
        raise gr.Error("Пожалуйста, загрузите ЕДИНСТВЕННЫЙ аудио-референс.")
    
    print(f"Получен текст: '{text}'")
    
    # --- Подготовка референс-аудио ---
    sr, arr = reference_audio
    
    # Конвертируем в float, если нужно
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 32767.0
    # Нормализуем громкость
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(arr)
    normalized_arr = pyln.normalize.loudness(arr, loudness, -23.0)

    # Сохраняем обработанный референс
    ref_path = os.path.join(TEMP_DIR, "processed_reference.wav")
    write_wav(ref_path, sr, (normalized_arr * 32767).astype(np.int16))
    print(f"Референс-аудио обработан и сохранен: {ref_path}")

    # --- Этап 1: Генерация базы с помощью XTTS на максималках ---
    print("Этап 1: Генерация с помощью XTTS...")
    base_speech_path = os.path.join(TEMP_DIR, 'base_xtts_speech.wav')
    
    # --- НАСТРОЙКИ КАЧЕСТВА XTTS ---
    # Эти параметры заставляют XTTS работать "дольше и лучше", как ты и хотел
    tts.tts_to_file(
        text=text,
        file_path=base_speech_path,
        language="ru",
        speaker_wav=ref_path,
        temperature=0.7,        # Более креативный и разнообразный результат (стандарт 0.65)
        length_penalty=1.0,     # Без штрафов за длину
        repetition_penalty=10.0, # Сильно штрафуем за повторения, убирает "заикания"
        top_k=50,
        top_p=0.85,
    )
    print("База XTTS сгенерирована.")

    # --- Этап 2: Клонирование с помощью OpenVoice на максималках ---
    print("Этап 2: Полировка с помощью OpenVoice...")
    
    # Извлекаем тембр цели (твой референс)
    target_se, _ = se_extractor.get_se(ref_path, tone_color_converter, target_dir=TEMP_DIR, vad=True)
    
    # Извлекаем тембр источника (то, что сгенерировал XTTS)
    source_se, _ = se_extractor.get_se(base_speech_path, tone_color_converter, target_dir=TEMP_DIR, vad=True)

    # --- НАСТРОЙКИ КАЧЕСТВА OPENVOICE ---
    output_path = os.path.join(OUTPUT_DIR, 'final_output.wav')
    tone_color_converter.convert(
        audio_src_path=base_speech_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message="@MyShell"
         
    )
    print("Голос успешно сгенерирован.")
    return base_speech_path, output_path

# --- 3. Интерфейс ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultimate Voice Clone: XTTS + OpenVoice (Max Quality & Speed)")
    gr.Markdown("Единый референс для интонации и тембра. Максимальное качество. Многопоточный рендеринг.")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст", placeholder="Введите текст на русском языке.", lines=4)
            audio_input = gr.Audio(label="Единственный аудио-референс (5-30 сек, WAV, без шума)", type="numpy")
            generate_button = gr.Button("Генерировать", variant="primary")
        with gr.Column(scale=1):
            xtts_output = gr.Audio(label="Выход XTTS (база)")
            final_output = gr.Audio(label="Выход OpenVoice (финал)")

    generate_button.click(
        fn=generate_voice,
        inputs=[text_input, audio_input],
        outputs=[xtts_output, final_output]
    )

# --- 4. Запуск ---
if __name__ == "__main__":
    demo.launch(share=True)
