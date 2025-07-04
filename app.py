import os
import torch
import numpy as np
import gradio as gr
from scipy.io.wavfile import write as write_wav
import warnings

from ruaccent import RUAccent
from TTS.api import TTS

warnings.filterwarnings("ignore")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
except ImportError as e:
    pass

# --------------------
# Константы и пути
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if DEVICE.type == "cuda" else "cpu"
OUTPUT_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
BASE_SE_DIR = "checkpoints/base_speakers/RU_SILERO"  # не используется, но пусть будет для совместимости
REF_TEXT = (
    "Прекрасная пора и майский день дождливый. Сижу у окна и чувствую аромат. "
    "Я вспоминаю с радостью сегодняшнюю встречу, и сердце бьётся в упоении. "
    "Слышу голос, смотрю на букет. Я в эйфории! Снова вдыхаю запах сирени. "
    "Вчитываюсь в письмо. После замираю. Ошибка в имени. Одна буква. Письмо сестре! "
    "Мгновенно выступившие слёзы."
)
SAMPLE_RATE = 24000  # XTTS v2 выдаёт 24кГц по умолчанию
LANGUAGE = "ru"

# --------------------
# Подготовка папок
# --------------------
for path in (OUTPUT_DIR, TEMP_DIR):
    os.makedirs(path, exist_ok=True)

# --------------------
# Загрузка моделей
# --------------------
print("🔄 Загрузка XTTS v2...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE_STR)

print("🔄 Загрузка RUAccent...")
accent_model = RUAccent()
accent_model.load(omograph_model_size="big_poetry", use_dictionary=True)

print("✅ Все модели готовы.")

# --------------------
# Функции генерации
# --------------------
def accent_text(text: str) -> str:
    """Расставляет ударения в тексте."""
    if not text.strip():
        return ""
    return accent_model.process_all(text)

def generate_voice_xtts(accented_text: str, reference_audio: tuple) -> str:
    """Генерирует аудио с клонированием голоса через XTTS v2."""
    if not accented_text:
        raise gr.Error("Введите текст для синтеза.")
    if reference_audio is None:
        raise gr.Error("Загрузите аудио-референс (5–15 сек, WAV без шума).")

    # Сохраняем временный референс
    sr, audio_np = reference_audio
    ref_path = os.path.join(TEMP_DIR, "user_reference.wav")

    # XTTS требует int16 WAV
    if audio_np.dtype != np.int16:
        audio_np = audio_np / np.max(np.abs(audio_np))
        write_wav(ref_path, sr, (audio_np * 32767).astype(np.int16))
    else:
        write_wav(ref_path, sr, audio_np)

    output_path = os.path.join(OUTPUT_DIR, "final_output.wav")
    tts.tts_to_file(
        text=accented_text,
        file_path=output_path,
        speaker_wav=ref_path,
        language=LANGUAGE,
        split_sentences=True,
        speed=1.0
    )
    return output_path

# --------------------
# Gradio UI
# --------------------
with gr.Blocks(title="OpenVoice RU (XTTS v2)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## XTTS v2 Voice Clone (Русский)")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст для синтеза", placeholder="Введите текст...")
            accent_button = gr.Button("Расставить ударения")
            accented_output = gr.Textbox(label="Текст с ударениями", interactive=True)
            audio_input = gr.Audio(label="Аудио-референс (WAV, 5–15 сек)", type="numpy")
            gen_button = gr.Button("Генерировать", variant="primary")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Синтезированный звук")

    accent_button.click(accent_text, inputs=text_input, outputs=accented_output)
    gen_button.click(generate_voice_xtts, inputs=[accented_output, audio_input], outputs=audio_output)

if __name__ == "__main__":
    demo.launch(
        share=True,
        enable_queue=True,
        server_name="0.0.0.0",
        server_port=7860
    )
