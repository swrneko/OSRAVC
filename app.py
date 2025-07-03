import gradio as gr
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# --- 1. Настройка и загрузка моделей ---
# Это выполнится один раз при запуске скрипта

print("Загрузка моделей OpenVoice...")

# Указываем устройство. Начнем с CPU.
# Когда решим проблему с ROCm, здесь можно будет написать "cuda:0"
device = "cpu"

# Создаем папку для выходных файлов, если ее нет
os.makedirs("outputs", exist_ok=True)

# Загружаем базовую модель Text-to-Speech
base_speaker_tts = BaseSpeakerTTS(f'checkpoints/base_speakers/EN/config.json', device=device)
base_speaker_tts.load_ckpt(f'checkpoints/base_speakers/EN/checkpoint.pth')

# Загружаем модель для конвертации тона и цвета голоса
tone_color_converter = ToneColorConverter(f'checkpoints/converter/config.json', device=device)
tone_color_converter.load_ckpt(f'checkpoints/converter/checkpoint.pth')

print("Модели успешно загружены.")

# --- 2. Основная функция для генерации ---
# Эта функция будет вызываться каждый раз, когда пользователь нажимает "Генерировать"

def generate_voice(text_to_speak, reference_audio, voice_style):
    if not text_to_speak:
        raise gr.Error("Пожалуйста, введите текст для озвучивания.")
    if reference_audio is None:
        raise gr.Error("Пожалуйста, загрузите аудиофайл-референс.")

    print("Генерация голоса...")
    
    # Путь, куда будет сохранен результат
    output_path = 'outputs/generated_voice.wav'
    
    # Извлекаем "цвет" голоса (speaker embedding) из аудио-референса
    # target_se - это тензор, представляющий уникальные характеристики голоса
    # save_path - временный файл, куда сохранится эмбеддинг
    reference_speaker = reference_audio
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='outputs/temp_voice_colors', vad=True)

    # --- Генерация голоса в два этапа ---
    
    # Этап 1: Генерируем речь с голосом по умолчанию (EN-US Male)
    # Это будет "сырая" речь с правильным произношением
    source_se = torch.load(f'checkpoints/base_speakers/EN/en_default_se.pth').to(device)
    base_speaker_tts.tts(text_to_speak, output_path, speaker='default', language='English', speed=1.0)

    # Этап 2: "Окрашиваем" сгенерированную речь голосом из референса
    # Мы берем речь из Шага 1 и накладываем на нее тембр из нашего аудиофайла
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=output_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message=encode_message)

    print(f"Голос успешно сгенерирован и сохранен в {output_path}")
    
    # Возвращаем путь к файлу, чтобы Gradio мог его воспроизвести
    return output_path

# --- 3. Создание интерфейса Gradio ---

with gr.Blocks(title="OpenVoice UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OpenVoice: Клонирование голоса")
    gr.Markdown("Введите текст, загрузите короткое аудио с голосом-примером (WAV, MP3), и модель сгенерирует речь этим голосом.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Текст для озвучивания", placeholder="Hello, this is a test of the OpenVoice system.")
            audio_input = gr.Audio(label="Аудиофайл-референс (3-10 секунд)", type="filepath")
            style_input = gr.Radio(
                ["Default"], # В будущем можно добавить стили, пока оставим один
                label="Стиль голоса",
                value="Default"
            )
            generate_button = gr.Button("Генерировать", variant="primary")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Результат")
            gr.Markdown("Пример референса можно взять [здесь](https://github.com/myshell-ai/OpenVoice/blob/main/resources/demo_speaker1.mp3).")

    generate_button.click(
        fn=generate_voice,
        inputs=[text_input, audio_input, style_input],
        outputs=[audio_output]
    )

# --- 4. Запуск приложения ---
if __name__ == "__main__":
    demo.launch(share=True) # share=True создает временную публичную ссылку
