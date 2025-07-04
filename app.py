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

# Настройка многопоточности PyTorch для ускорения на CPU
if torch.get_num_threads() > 1:
    try:
        # Устанавливаем количество потоков равным количеству ядер CPU
        torch.set_num_threads(os.cpu_count())
        print(f"🔥 Установлено {os.cpu_count()} потоков для PyTorch.")
    except Exception as e:
        print(f"Не удалось установить количество потоков: {e}")


print("Инициализация...")
DEVICE = "cpu"
OUTPUT_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs('checkpoints/base_speakers/RU_SILERO', exist_ok=True) # Папка для эталона Silero

# --- 1. Загрузка моделей ---
import silero
from ruaccent import RUAccent
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

print("🔄 Загрузка Silero TTS...")
# Выбираем стабильный и качественный голос
SPEAKER = 'baya' 
LANGUAGE = 'ru'
SAMPLE_RATE = 48000
# Используем torch.hub.set_dir, чтобы избежать проблем с кэшем
torch.hub.set_dir('.') 
silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=LANGUAGE, speaker=f'v3_1_{LANGUAGE}')
silero_model.to(DEVICE)


print("🔄 Загрузка OpenVoice Converter...")
tone_color_converter = ToneColorConverter('checkpoints/converter/config.json', device=DEVICE)
tone_color_converter.load_ckpt('checkpoints/converter/checkpoint.pth')

print("🔄 Загрузка RUAccent...")
accent_model = RUAccent()
accent_model.load(omograph_model_size="big_poetry", use_dictionary=True)

# Создание/загрузка эталонного тембра для базового голоса Silero
source_se_path = f'checkpoints/base_speakers/RU_SILERO/ru_{SPEAKER}_se.pth'
if not os.path.exists(source_se_path):
    print(f"✨ Генерация эталонного тембра для `{SPEAKER}`...")
    ref_text = "Здравствуйте, это стандартный образец голоса для системы клонирования тембра."
    ref_wav = os.path.join(TEMP_DIR, "reference_silero.wav")
    silero_model.save_wav(text=ref_text, speaker=SPEAKER, sample_rate=SAMPLE_RATE, audio_path=ref_wav)
    se, _ = se_extractor.get_se(ref_wav, tone_color_converter, target_dir=TEMP_DIR, vad=True)
    torch.save(se, source_se_path)
    print(f"✅ Эталонный тембр сохранён: {source_se_path}")

source_se = torch.load(source_se_path, map_location=DEVICE)
print("✅ Все модели и эталонный тембр готовы.")


# --- 2. Функционал ---
def accent_text(text: str) -> str:
    """Расставляет ударения в тексте."""
    return accent_model.process_all(text) if text.strip() else ""

def generate_voice(accented_text: str, reference_audio: tuple):
    if not accented_text.strip():
        raise gr.Error("Пожалуйста, введите текст.")
    if reference_audio is None:
        raise gr.Error("Пожалуйста, загрузите ЕДИНСТВЕННЫЙ аудио-референс.")
    
    print(f"Получен текст: '{accented_text}'")
    
    # --- Подготовка референс-аудио ---
    sr, arr = reference_audio
    
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 32767.0
    
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(arr)
    normalized_arr = pyln.normalize.loudness(arr, loudness, -23.0)

    ref_path = os.path.join(TEMP_DIR, "processed_reference.wav")
    write_wav(ref_path, sr, (normalized_arr * 32767).astype(np.int16))
    print(f"Референс-аудио обработан и сохранен: {ref_path}")

    # --- Этап 1: Генерация базы с помощью Silero ---
    print("Этап 1: Генерация с помощью Silero...")
    base_speech_path = os.path.join(TEMP_DIR, 'base_silero_speech.wav')
    silero_model.save_wav(text=accented_text, speaker=SPEAKER, sample_rate=SAMPLE_RATE, audio_path=base_speech_path)
    print("База Silero сгенерирована.")

    # --- Этап 2: Клонирование с помощью OpenVoice ---
    print("Этап 2: Полировка с помощью OpenVoice...")
    
    target_se, _ = se_extractor.get_se(ref_path, tone_color_converter, target_dir=TEMP_DIR, vad=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'final_output.wav')
    tone_color_converter.convert(
        audio_src_path=base_speech_path, 
        src_se=source_se, # Используем наш загруженный эталон
        tgt_se=target_se, 
        output_path=output_path,
        message="@MyShell"
    )
    print("Голос успешно сгенерирован.")
    return base_speech_path, output_path

# --- 3. Интерфейс ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ultimate Voice Clone: Silero + OpenVoice")
    gr.Markdown("Единый референс. Максимальная стабильность. Многопоточный рендеринг.")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст", placeholder="Введите текст на русском языке.", lines=4)
            accent_button = gr.Button("Расставить ударения")
            accented_text_output = gr.Textbox(label="Текст с ударениями", interactive=True)
            audio_input = gr.Audio(label="Единственный аудио-референс (5-30 сек, WAV, без шума)", type="numpy")
            generate_button = gr.Button("Генерировать", variant="primary")
        with gr.Column(scale=1):
            silero_output = gr.Audio(label="Выход Silero (база)")
            final_output = gr.Audio(label="Выход OpenVoice (финал)")

    accent_button.click(fn=accent_text, inputs=text_input, outputs=accented_text_output)
    generate_button.click(
        fn=generate_voice,
        inputs=[accented_text_output, audio_input],
        outputs=[silero_output, final_output]
    )

# --- 4. Запуск ---
if __name__ == "__main__":
    demo.launch(share=True, enable_queue=True)
