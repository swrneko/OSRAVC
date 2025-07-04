
import os
import uuid
import torch
import numpy as np
import gradio as gr
from scipy.io.wavfile import write as write_wav
import warnings

from ruaccent import RUAccent
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


# --------------------
# Константы и пути (укажи свои пути здесь)
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if DEVICE.type == "cuda" else "cpu"
# Путь к локальной XTTS v2 модели
XTTS_MODEL_PATH = "checkpoints/xttsv2_custom"
# Параметры OpenVoice
CONVERTER_CFG = "checkpoints/converter/config.json"
CONVERTER_CKPT = "checkpoints/converter/checkpoint.pth"
# Выходные директории
OUTPUT_DIR = "outputs"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
TEMP_XTTS = os.path.join(TEMP_DIR, "xtts")
TEMP_COLORS = os.path.join(TEMP_DIR, "colors")
SE_DIR = os.path.join(OUTPUT_DIR, "se_refs")  # для хранения .pth
# Общий текст, если нужен
REF_TEXT = "Прекрасная пора и майский день дождливый..."
LANGUAGE = "ru"
SAMPLE_RATE = 24000

# --------------------
# Подготовка папок
# --------------------
for d in (OUTPUT_DIR, TEMP_DIR, TEMP_XTTS, TEMP_COLORS, SE_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------
# Загрузка ранее сохранённых SE
# --------------------
se_store = {}
for f in os.listdir(SE_DIR):
    if f.endswith(".pth"):
        name = f[:-4]
        se_store[name] = torch.load(os.path.join(SE_DIR, f), map_location=DEVICE)

# --------------------
# Загрузка моделей
# --------------------
print("🔄 Загружаем XTTS v2...")
tts = TTS(model_name=XTTS_MODEL_PATH).to(DEVICE_STR)

print("🔄 Загружаем OpenVoice Converter...")
converter = ToneColorConverter(CONVERTER_CFG, device=DEVICE_STR)
converter.load_ckpt(CONVERTER_CKPT)

print("🔄 Загружаем RUAccent...")
accent = RUAccent()
accent.load(omograph_model_size="big_poetry", use_dictionary=True)

# --------------------
# Функции
# --------------------

def accent_text(txt: str) -> str:
    return accent.process_all(txt) if txt.strip() else ""


def create_se_from_ref(ref: tuple) -> str:
    if ref is None:
        raise gr.Error("Загрузите аудио для эталона.")
    sr, arr = ref
    arr = arr / np.max(np.abs(arr))
    uid = str(uuid.uuid4())[:8]
    wav_path = os.path.join(SE_DIR, f"se_{uid}.wav")
    write_wav(wav_path, sr, (arr * 32767).astype(np.int16))
    se, _ = se_extractor.get_se(wav_path, converter, target_dir=TEMP_COLORS, vad=False)
    se_store[uid] = se
    torch.save(se, os.path.join(SE_DIR, f"{uid}.pth"))
    return f"Эталон создан: {uid}", list(se_store.keys())


def generate(accented: str, user_ref: tuple, se_id: str):
    if not accented.strip():
        raise gr.Error("Введите текст.")
    if user_ref is None:
        raise gr.Error("Загрузите референс.")
    if se_id not in se_store:
        raise gr.Error("Выберите SE из списка.")

    # XTTS
    xtts_path = os.path.join(TEMP_XTTS, "out_xtts.wav")
    sr, arr = user_ref
    arr = arr / np.max(np.abs(arr))
    tmp_ref = os.path.join(TEMP_COLORS, "tmp_ref.wav")
    write_wav(tmp_ref, sr, (arr * 32767).astype(np.int16))
    tts.tts_to_file(text=accented, file_path=xtts_path,
                    speaker_wav=tmp_ref, language=LANGUAGE,
                    split_sentences=True, speed=1.0)

    # OpenVoice
    tgt_se, _ = se_extractor.get_se(xtts_path, converter, target_dir=TEMP_COLORS, vad=False)
    final_path = os.path.join(OUTPUT_DIR, "final.wav")
    converter.convert(audio_src_path=xtts_path,
                      src_se=se_store[se_id], tgt_se=tgt_se,
                      output_path=final_path,
                      message="@MyShell",
                      use_griffin_lim=False,
                      spectrogram_refinement=True)
    return xtts_path, final_path

# --------------------
# UI
# --------------------
with gr.Blocks(title="XTTSv2 + OpenVoice RU") as demo:
    gr.Markdown("## Создать эталон из референса")
    with gr.Row():
        upload_se = gr.Audio(label="Референс для SE (WAV)", type="numpy")
        btn_se = gr.Button("Сгенерировать SE")
        status = gr.Textbox(label="Статус")
        ddl = gr.Dropdown(label="SE ID", choices=list(se_store.keys()))
        btn_se.click(create_se_from_ref, inputs=upload_se, outputs=[status, ddl])

    gr.Markdown("## Генерация речи")
    with gr.Row():
        txt_in = gr.Textbox(label="Текст")
        btn_acc = gr.Button("Ударения")
        txt_acc = gr.Textbox(label="Текст с ударениями")
        upload_ref = gr.Audio(label="Референс для XTTS", type="numpy")
        btn_gen = gr.Button("Генерировать")
    with gr.Row():
        out_xtts = gr.Audio(label="XTTS Audio", type="filepath")
        out_final = gr.Audio(label="OpenVoice Audio", type="filepath")

    btn_acc.click(accent_text, inputs=txt_in, outputs=txt_acc)
    btn_gen.click(generate, inputs=[txt_acc, upload_ref, ddl], outputs=[out_xtts, out_final])

if __name__ == "__main__":
    demo.launch(share=True, enable_queue=True, server_name="0.0.0.0")

