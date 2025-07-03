import gradio as gr
import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
import warnings
import silero
from ruaccent import RUAccent
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

warnings.filterwarnings("ignore")

print("Загрузка моделей...")
device = "cpu"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs('outputs/temp_colors', exist_ok=True)
os.makedirs('checkpoints/base_speakers/RU_SILERO', exist_ok=True)

tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

language = 'ru'
model_id = 'v4_ru'
speaker = 'baya'
silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language, speaker=model_id)
silero_model.to(device)

accent_model = RUAccent()
accent_model.load(omograph_model_size='big_poetry', use_dictionary=True)

source_se_path = 'checkpoints/base_speakers/RU_XTTS/ru_xtts_se.pth'
if not os.path.exists(source_se_path):
    print("Создание эталонного русского тембра...")
    ref_text = "Здравствуйте, это стандартный образец голоса для системы клонирования тембра."
    ref_wav = os.path.join(output_dir, "ru_reference.wav")
    silero_model.save_wav(text=ref_text, speaker=speaker, sample_rate=48000, audio_path=ref_wav)
    se, _ = se_extractor.get_se(ref_wav, tone_color_converter, target_dir='outputs/temp_colors', vad=True)
    torch.save(se, source_se_path)
    print(f"Эталонный тембр для голоса '{speaker}' создан: {source_se_path}")

source_se = torch.load(source_se_path).to(device)
print("Все модели успешно загружены.")

def accent_text(text):
    if not text: return ""
    return accent_model.process_all(text)

def generate_russian_voice(accented_text, reference_audio_data):
    if not accented_text: raise gr.Error("Введите текст.")
    if reference_audio_data is None: raise gr.Error("Загрузите аудио-референс.")
    
    ref_sample_rate, ref_audio_array = reference_audio_data
    temp_reference_path = os.path.join(output_dir, "temp_reference.wav")
    write_wav(temp_reference_path, ref_sample_rate, ref_audio_array.astype(np.int16))

    base_speech_path = os.path.join(output_dir, 'base_russian_speech.wav')
    silero_model.save_wav(text=accented_text, speaker=speaker, sample_rate=48000, audio_path=base_speech_path)

    target_se, _ = se_extractor.get_se(temp_reference_path, tone_color_converter, target_dir='outputs/temp_colors', vad=True)

    output_path = os.path.join(output_dir, 'generated_russian_voice.wav')
    tone_color_converter.convert(audio_src_path=base_speech_path, src_se=source_se, tgt_se=target_se, output_path=output_path, message="@MyShell")
    return output_path

with gr.Blocks(title="Russian OpenVoice UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OpenVoice: Клонирование голоса на русском языке")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст", placeholder="Введите текст на русском языке.")
            accent_button = gr.Button("Расставить ударения", variant="secondary")
            accented_text_output = gr.Textbox(label="Проверка ударений", interactive=True)
            audio_input = gr.Audio(label="Аудио-референс", type="numpy")
            generate_button = gr.Button("Генерировать", variant="primary")
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Результат")
    accent_button.click(fn=accent_text, inputs=[text_input], outputs=[accented_text_output])
    generate_button.click(fn=generate_russian_voice, inputs=[accented_text_output, audio_input], outputs=[audio_output])

if __name__ == "__main__":
    demo.launch(share=True)
