# webrtc-vad-gradio
This project uses the python library Gradio to implement the online vocal detection function of webrtc-vad, which mainly involves engineering the compiled webrtc-vad file using python. You can run this project directly to achieve the purpose of online voice detection, or you can use the compiled webrtc-vad file provided by this project to complete the work you want to do.

# instruction
1. The model supports input audio in wav, mp3 and other formats, because the backend will be processed to pcm format through ffmpeg unification, this is because the webrtc-vad project is to support pcm format.
2. Because webrtc-vad is based on sound energy thresholds for judgment, the model becomes less effective in high noise environments.
3. The model supports passing in two parameters, the length of the audio to be recognized and the aggressiveness of the model recognition. The length of the audio to be recognized is 10, 20, and 30 milliseconds, and the degree of aggressiveness is 0, 1, 2, and 3.
4. The model cuts the audio into small segments, each of which is the size of the length of the audio to be detected, and then detects each segment, and finally splices the output of the audio with and without human voices, respectively.

# gradio-generated interface display
![image](https://github.com/xinliu9451/webrtc-vad-gradio/blob/main/image/gradio.png)

# reference
https://github.com/wiseman/py-webrtcvad
