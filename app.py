import gradio as gr
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import numpy as np
from scipy.io.wavfile import write as write_wav

# --- 1. Настройка и загрузка моделей ---

print("Загрузка моделей OpenVoice...")
device = "cpu"
os.makedirs("outputs", exist_ok=True)

base_speaker_tts = BaseSpeakerTTS(f'checkpoints/base_speakers/EN/config.json', device=device)
base_speaker_tts.load_ckpt(f'checkpoints/base_speakers/EN/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

print("Модели успешно загружены.")

# --- 2. Обновленная основная функция для генерации ---
def generate_voice(text_to_speak, reference_audio_data):
    if not text_to_speak:
        raise gr.Error("Пожалуйста, введите текст для озвучивания.")
    if reference_audio_data is None:
        raise gr.Error("Пожалуйста, загрузите или запишите аудио-референс.")

    print("Обработка референса и генерация голоса...")
    
    sample_rate, audio_array = reference_audio_data
    
    temp_reference_path = 'outputs/temp_reference.wav'
    write_wav(temp_reference_path, sample_rate, audio_array.astype(np.int16))

    output_path = 'outputs/generated_voice.wav'
    
    target_se, audio_name = se_extractor.get_se(temp_reference_path, tone_color_converter, target_dir='outputs/temp_voice_colors', vad=True)

    source_se = torch.load(f'checkpoints/base_speakers/EN/en_default_se.pth').to(device)
    base_speaker_tts.tts(text_to_speak, output_path, speaker='default', language='English', speed=1.0)

    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=output_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message=encode_message)

    print(f"Голос успешно сгенерирован и сохранен в {output_path}")
    
    return output_path

# --- 3. Обновленный интерфейс Gradio ---

with gr.Blocks(title="OpenVoice UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OpenVoice: Клонирование голоса с записью и обрезкой")
    gr.Markdown("Введите текст, затем **загрузите аудиофайл** или **запишите голос с микрофона**. В загруженном аудио вы можете выделить нужный фрагмент (3-10 секунд).")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст для озвучивания", placeholder="Hello, this is a test of the OpenVoice system.")
            
            audio_input = gr.Audio(
                label="Аудиофайл-референс",
                sources=["upload", "microphone"], 
                type="numpy" 
            )
            
            generate_button = gr.Button("Генерировать", variant="primary")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Результат")
            gr.Markdown("Пример референса можно взять [здесь](https://github.com/myshell-ai/OpenVoice/blob/main/resources/demo_speaker1.mp3).")

    generate_button.click(
        fn=generate_voice,
        inputs=[text_input, audio_input],
        outputs=[audio_output]
    )

# --- 4. Запуск приложения ---
if __name__ == "__main__":
    demo.launch(share=True) # Оставляем share=True для надежности
