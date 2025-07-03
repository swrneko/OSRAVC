import gradio as gr
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import numpy as np
from scipy.io.wavfile import write as write_wav # Импортируем функцию для записи WAV

# --- 1. Настройка и загрузка моделей (без изменений) ---

print("Загрузка моделей OpenVoice...")
device = "cpu"
os.makedirs("outputs", exist_ok=True)

base_speaker_tts = BaseSpeakerTTS(f'checkpoints/base_speakers/EN/config.json', device=device)
base_speaker_tts.load_ckpt(f'checkpoints/base_speakers/EN/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

print("Модели успешно загружены.")


# --- 2. Обновленная основная функция для генерации ---

def generate_voice(text_to_speak, reference_audio_data, voice_style):
    # Проверки на наличие входных данных
    if not text_to_speak:
        raise gr.Error("Пожалуйста, введите текст для озвучивания.")
    if reference_audio_data is None:
        raise gr.Error("Пожалуйста, загрузите или запишите аудио-референс.")

    print("Обработка референса и генерация голоса...")

    # ИЗМЕНЕНИЕ: Обрабатываем данные, полученные от Gradio
    # Gradio возвращает кортеж (частота дискретизации, numpy-массив с аудио)
    sample_rate, audio_array = reference_audio_data
    
    # Сохраняем записанное/обрезанное аудио во временный файл
    temp_reference_path = 'outputs/temp_reference.wav'
    # Убедимся, что данные в формате 16-bit integer, как ожидает большинство WAV-файлов
    write_wav(temp_reference_path, sample_rate, audio_array.astype(np.int16))

    # Далее код работает как и раньше, но использует путь к нашему временному файлу
    output_path = 'outputs/generated_voice.wav'
    
    # Извлекаем "цвет" голоса из нашего временного файла
    target_se, audio_name = se_extractor.get_se(temp_reference_path, tone_color_converter, target_dir='outputs/temp_voice_colors', vad=True)

    # Этап 1: Генерация базовой речи
    source_se = torch.load(f'checkpoints/base_speakers/EN/en_default_se.pth').to(device)
    base_speaker_tts.tts(text_to_speak, output_path, speaker='default', language='English', speed=1.0)

    # Этап 2: "Окрашивание" речи
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
            
            # ИЗМЕНЕНИЕ: Обновляем компонент gr.Audio
            audio_input = gr.Audio(
                label="Аудиофайл-референс",
                sources=["upload", "microphone"], # Добавляем запись с микрофона
                type="numpy" # Меняем тип, чтобы получать данные для обрезки
            )
            
            style_input = gr.Radio(
                ["Default"],
                label="Стиль голоса",
                value="Default"
            )
            generate_button = gr.Button("Генерировать", variant="primary")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Результат")
            gr.Markdown("Пример референса можно взять [здесь](https://github.com/myshell-ai/OpenVoice/blob/main/resources/demo_speaker1.mp3).")

    # .click() остается без изменений, Gradio сам передаст данные в нужном формате
    generate_button.click(
        fn=generate_voice,
        inputs=[text_input, audio_input, style_input],
        outputs=[audio_output]
    )

# --- 4. Запуск приложения (без изменений) ---
if __name__ == "__main__":
    demo.launch(share=True)
