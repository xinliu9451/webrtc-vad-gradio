import gradio as gr

import subprocess
import numpy as np
import librosa
import re
import os


def run_vad_detection(audio_file, max_frames, vad_mode, frame_duration):
    # 运行编译好的程序
    command = f"./myprogram {audio_file} {max_frames} {vad_mode} {frame_duration}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # 将输出解码为字符串
    output = stdout.decode('utf-8')
    # 提取结果部分
    results_section = output.split('所有音频片段检测结果：\n')[-1]
    
    # 初始化结果列表
    timestamps = []
    is_speech_results = []
    
    # 解析每一行结果
    for line in results_section.strip().split('\n'):
        # 使用正则表达式提取时间戳和语音检测结果
        match = re.search(r'音频片段ID: ([\d.]+), is_speech: (\w+)', line)
        if match:
            timestamp = float(match.group(1))
            is_speech = True if match.group(2).startswith('True') else False
            
            timestamps.append(timestamp)
            is_speech_results.append(is_speech)
    
    return timestamps, is_speech_results

def calculate_max_frames(audio_file, frame_duration):
    with open(audio_file, 'rb') as pcm_f:
        pcm_data = pcm_f.read()

    # 转换为16位有符号整型数组
    audio = np.frombuffer(pcm_data, dtype=np.int16)

    # 将数据归一化到-1到1范围
    audio = audio.astype(np.float32) / 32768.0
    max_frames = int((len(audio) / 16000) * (1000 / frame_duration))
    
    return max_frames

def split_and_merge_audio(wav_file, timestamps, labels):
    # 使用 librosa 以 16,000 Hz 采样率加载音频
    audio_data, sr = librosa.load(wav_file, sr=16000, mono=True)

    # 转换时间段为采样点索引
    time_intervals = [(int(start * sr), int(end * sr)) 
                      for start, end in zip([0] + timestamps[:-1], timestamps)]

    # 用于存储分段音频
    voiced_segments = []
    unvoiced_segments = []

    # 遍历时间段和检测结果
    for (start_idx, end_idx), is_voiced in zip(time_intervals, labels):
        segment = audio_data[start_idx:end_idx]
        if is_voiced:
            voiced_segments.append(segment)
        else:
            unvoiced_segments.append(segment)

    # 拼接片段
    voiced_audio = np.concatenate(voiced_segments) if voiced_segments else np.array([], dtype=np.float32)
    unvoiced_audio = np.concatenate(unvoiced_segments) if unvoiced_segments else np.array([], dtype=np.float32)

    return voiced_audio, unvoiced_audio, audio_data


html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Introduce</h2>
    <p style="font-size: 18px;margin-left: 20px;">这是一个基于webrtc-vad的人声检测Demo，它是基于声音能量阈值的方式进行检测，它会将音频切割为小段，每段长度为待检测音频长度的大小，然后检测每段音频，最后将有人声和无人声的音频分别进行拼接输出。</p>
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2>
    <p style="font-size: 18px;margin-left: 20px;">支持上传wav、mp3格式的音频。</p>
    <p style="font-size: 18px;margin-left: 20px;">在高噪声的环境下会对识别结果产生一定的影响。</p>
</div>
"""


def model_inference(audio_input, vad_mode, frame_duration):
    pcm_audio_path = audio_input[:-3] + 'pcm'
    subprocess.run(["ffmpeg", "-i", audio_input, "-ar", "16000", "-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le", pcm_audio_path, "-y"], check=True)
    max_frames = calculate_max_frames(pcm_audio_path, frame_duration)
    timestamps, is_speech = run_vad_detection(pcm_audio_path, max_frames, vad_mode, frame_duration)
    voiced_audio, unvoiced_audio , audio_data= split_and_merge_audio(audio_input, timestamps, is_speech)
    os.remove(pcm_audio_path)
    return (16000, voiced_audio), (16000, unvoiced_audio), {"timestamps": timestamps, "is_speech": is_speech}

def launch():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML(html_content)
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Configuration"):
                    vad_mode = gr.Dropdown(choices=[0, 1, 2, 3], value=2, label="模型激进程度，数值越大越激进")
                    frame_duration = gr.Dropdown(choices=[10, 20, 30], value=30, label="待检测音频时长(毫秒)")
                with gr.Row():
                    audio_input = gr.Audio(type="filepath", label="仅支持wav格式")

                fn_button = gr.Button("Start", variant="primary")
                audio_outputs1 = gr.Audio(label="Separated Speech")
                audio_outputs2 = gr.Audio(label="Separated NO-Speech")
                text_outputs = gr.Textbox(label="Inference results")

        fn_button.click(model_inference, inputs=[audio_input, vad_mode, frame_duration], outputs=[audio_outputs1, audio_outputs2, text_outputs])

    demo.launch(share=True)


if __name__ == "__main__":
    launch()