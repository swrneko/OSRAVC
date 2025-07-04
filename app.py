import os
import torch
import numpy as np
import gradio as gr
from scipy.io.wavfile import write as write_wav
import warnings

from ruaccent import RUAccent
from silero import silero_tts
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

warnings.filterwarnings("ignore")

# --------------------
# Константы и пути
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if DEVICE.type == "cuda" else "cpu"
OUTPUT_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
TEMP_COLORS = os.path.join(TEMP_DIR, "colors")
BASE_SE_DIR = "checkpoints/base_speakers/RU_SILERO"
CONVERTER_CFG = "checkpoints/converter/config.json"
CONVERTER_CKPT = "checkpoints/converter/checkpoint.pth"
REF_TEXT = (
    "Прекрасная пора и майский день дождливый. Сижу у окна и чувствую аромат. "
    "Я вспоминаю с радостью сегодняшнюю встречу, и сердце бьётся в упоении. "
    "Слышу голос, смотрю на букет. Я в эйфории! Снова вдыхаю запах сирени. "
    "Вчитываюсь в письмо. После замираю. Ошибка в имени. Одна буква. Письмо сестре! "
    "Мгновенно выступившие слёзы."
)
SAMPLE_RATE = 48000
SPEAKER = "kseniya"
LANGUAGE = "ru"

# --------------------
# Подготовка папок
# --------------------
for path in (OUTPUT_DIR, TEMP_DIR, TEMP_COLORS, BASE_SE_DIR):
    os.makedirs(path, exist_ok=True)

# --------------------
# Загрузка моделей
# --------------------
print("🔄 Загрузка ToneColorConverter...")
tone_color_converter = ToneColorConverter(CONVERTER_CFG, device=DEVICE_STR)
tone_color_converter.load_ckpt(CONVERTER_CKPT)

print("🔄 Загрузка Silero TTS...")
hub_out = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language=LANGUAGE,
    speaker=f"v3_1_{LANGUAGE}",
    device=DEVICE_STR
)
silero_model = hub_out[0] if isinstance(hub_out, (tuple, list)) else hub_out

print("🔄 Загрузка RUAccent...")
accent_model = RUAccent()
accent_model.load(omograph_model_size="big_poetry", use_dictionary=True)

# --------------------
# Подготовка эталонного тембра
# --------------------
source_se_path = os.path.join(BASE_SE_DIR, f"ru_{SPEAKER}_se.pth")
if not os.path.exists(source_se_path):
    print(f"✨ Генерация эталонного тембра для `{SPEAKER}`...")
    ref_wav = os.path.join(TEMP_DIR, "reference_silero.wav")
    silero_model.save_wav(
        text=REF_TEXT,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE,
        audio_path=ref_wav
    )
    se, _ = se_extractor.get_se(
        ref_wav,
        tone_color_converter,
        target_dir=TEMP_COLORS,
        vad=True
    )
    torch.save(se, source_se_path)
    print(f"✅ Эталонный тембр сохранён: {source_se_path}")

source_se = torch.load(source_se_path, map_location=DEVICE).to(DEVICE)
print("✅ Все модели и эталонный тембр готовы.")

# --------------------
# Функции генерации
# --------------------
def accent_text(text: str) -> str:
    """Расставляет ударения в тексте."""
    if not text.strip():
        return ""
    return accent_model.process_all(text)

def generate_voice(accented_text: str, reference_audio: tuple) -> str:
    """Генерирует аудио с клонированием голоса."""
    if not accented_text:
        raise gr.Error("Введите текст для синтеза.")
    if reference_audio is None:
        raise gr.Error("Загрузите аудио-референс (5–15 сек, WAV без шума).")

    # Сохранение временного референса
    sr, audio_np = reference_audio
    ref_path = os.path.join(TEMP_DIR, "user_reference.wav")

    if audio_np.dtype != np.int16:
        audio_np = audio_np / np.max(np.abs(audio_np))
        write_wav(ref_path, sr, (audio_np * 32767).astype(np.int16))
    else:
        write_wav(ref_path, sr, audio_np)

    # Генерация базовой речи через Silero
    base_path = os.path.join(TEMP_DIR, "base_speech.wav")
    silero_model.save_wav(
        text=accented_text,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE,
        audio_path=base_path
    )

    # Извлечение целевого spectral envelope
    tgt_se, _ = se_extractor.get_se(
        ref_path,
        tone_color_converter,
        target_dir=TEMP_COLORS,
        vad=True
    )

    # Конвертация тембра
    output_path = os.path.join(OUTPUT_DIR, "final_output.wav")
    tone_color_converter.convert(
        audio_src_path=base_path,
        src_se=source_se,
        tgt_se=tgt_se,
        output_path=output_path,
        message="@MyShell"
    )

    return output_path

# --------------------
# Gradio UI
# --------------------
with gr.Blocks(title="OpenVoice RU", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## OpenVoice Clone (Русский)")

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
    gen_button.click(generate_voice, inputs=[accented_output, audio_input], outputs=audio_output)

if __name__ == "__main__":
    demo.launch(
        share=True,
        enable_queue=True,
        server_name="0.0.0.0",
        server_port=7860
    )

