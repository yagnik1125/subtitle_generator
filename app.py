import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
# import googletrans 
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import tempfile
import wave
import io
import subprocess
import cv2
import ffmpeg
import copy


# Initialize the Groq client
client = Groq(api_key="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")


# Streamlit frontend for audio input and translation
st.title("Audio Translation App")

# Audio file input
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac", "m4a"])
if uploaded_file:
    st.audio(uploaded_file, format="wav")
mic_audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🎙️ Stop Recording", key='recorder')
if mic_audio:
    st.write("mic audio through bytes")
    st.audio(mic_audio['bytes'], format='wav')
mic_audio_file_name='temp_mic_audio.wav'

vedio_file_name=""
uploaded_vedio_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
if uploaded_vedio_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_vedio_file.read())
    video_capture = cv2.VideoCapture(tfile.name)

    vedio_file_name = tfile.name
    st.write(f"Vedio File saved at: {vedio_file_name}")
    # st.text("Video Loaded Successfully!")
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    duration = total_frames / frame_rate

    # st.write(f"Total frames: {total_frames}")
    # st.write(f"Frame rate: {frame_rate} fps")
    # st.write(f"Duration: {duration} seconds")

    # st.video(uploaded_vedio_file)
    video_capture.release()
    # os.remove(tfile.name)

if mic_audio:
    # Get the byte data from the audio recorder
    audio_bytes = mic_audio['bytes']
    audio_file_like = io.BytesIO(audio_bytes)
    with wave.open(mic_audio_file_name, 'wb') as wav_file:
        sample_width = 2  # Sample width in bytes (16 bits)
        channels = 1      # Mono
        framerate = 44100 # Sample rate

        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        wav_file.writeframes(audio_bytes)
    # st.write("mic audio through wav")
    # st.audio(audio_file_like, format='wav')


def get_audio_buffer(audio_file):
    with open(audio_file, "rb") as f:
        audio_buffer = io.BytesIO(f.read())  # Read the file and store it in BytesIO buffer
    return audio_buffer

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"


def write_vtt(segments, file_path):
    with open(file_path, 'w', encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for i, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            text = segment['text']
            # Convert start and end times to VTT format (HH:MM:SS.mmm)
            start_time = "{:02}:{:02}:{:06.3f}".format(int(start // 3600), int((start % 3600) // 60), start % 60)
            end_time = "{:02}:{:02}:{:06.3f}".format(int(end // 3600), int((end % 3600) // 60), end % 60)
            vtt_file.write(f"{i}\n")
            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"{text}\n\n")

def translate_text(text, targ_lang):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"Translate this text to '{targ_lang}.' Translate the text even if it is in '{targ_lang}'. Text:{text}. ONLY RETURN TRANSLATED TEXT DO NOT WRITE ANYTHING ELSE",
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content 
        # print(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# def add_subtitles_to_video(input_video, subtitle_file, output_video):
#     # Use FFmpeg to add the subtitle to the video
#     command = [
#         'ffmpeg', '-y', '-i', input_video, '-vf', f"subtitles={subtitle_file}", output_video
#     ]
#     subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def add_subtitles_to_video(input_video, input_subtitle, output_video, font="Noto Sans Devanagari"):
    try:
        # Add subtitles filter with force_style option to specify font
        ffmpeg_output = (
            ffmpeg
            .input(input_video)
            .output(
                output_video,
                vf=f"subtitles={input_subtitle}:force_style='FontName={font}'",  # Applying subtitles with font style
                vcodec="libx264",  # Re-encode video to ensure filtering works
                acodec="aac",  # Re-encode audio
                strict="experimental"  # Required for AAC audio
            )
        )

        # Run the ffmpeg process
        ffmpeg_output.run(overwrite_output=True)
        print(f"Subtitles added successfully to {output_video} using font {font}")

    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")

def get_font_for_language(language):
    font_mapping = {
        'afrikaans': 'Noto Sans',
        'albanian': 'Noto Sans',
        'amharic': 'Noto Sans Ethiopic',
        'arabic': 'Noto Sans Arabic',
        'armenian': 'Noto Sans Armenian',
        'azerbaijani': 'Noto Sans',
        'basque': 'Noto Sans',
        'belarusian': 'Noto Sans',
        'bengali': 'Noto Sans Bengali',
        'bosnian': 'Noto Sans',
        'bulgarian': 'Noto Sans',
        'catalan': 'Noto Sans',
        'cebuano': 'Noto Sans',
        'chichewa': 'Noto Sans',
        'chinese (simplified)': 'Noto Sans CJK SC',
        'chinese (traditional)': 'Noto Sans CJK TC',
        'corsican': 'Noto Sans',
        'croatian': 'Noto Sans',
        'czech': 'Noto Sans',
        'danish': 'Noto Sans',
        'dutch': 'Noto Sans',
        'english': 'Noto Sans',
        'esperanto': 'Noto Sans',
        'estonian': 'Noto Sans',
        'filipino': 'Noto Sans',
        'finnish': 'Noto Sans',
        'french': 'Noto Sans',
        'frisian': 'Noto Sans',
        'galician': 'Noto Sans',
        'georgian': 'Noto Sans',
        'german': 'Noto Sans',
        'greek': 'Noto Sans Greek',
        'gujarati': 'Noto Sans Gujarati',
        'haitian creole': 'Noto Sans',
        'hausa': 'Noto Sans',
        'hawaiian': 'Noto Sans',
        'hebrew': 'Noto Sans Hebrew',
        'hindi': 'Noto Sans Devanagari',
        'hmong': 'Noto Sans',
        'hungarian': 'Noto Sans',
        'icelandic': 'Noto Sans',
        'igbo': 'Noto Sans',
        'indonesian': 'Noto Sans',
        'irish': 'Noto Sans',
        'italian': 'Noto Sans',
        'japanese': 'Noto Sans CJK JP',
        'javanese': 'Noto Sans',
        'kannada': 'Noto Sans Kannada',
        'kazakh': 'Noto Sans',
        'khmer': 'Noto Sans Khmer',
        'korean': 'Noto Sans CJK KR',
        'kurdish (kurmanji)': 'Noto Sans',
        'kyrgyz': 'Noto Sans',
        'lao': 'Noto Sans Lao',
        'latin': 'Noto Sans',
        'latvian': 'Noto Sans',
        'lithuanian': 'Noto Sans',
        'luxembourgish': 'Noto Sans',
        'macedonian': 'Noto Sans',
        'malagasy': 'Noto Sans',
        'malay': 'Noto Sans',
        'malayalam': 'Noto Sans Malayalam',
        'maltese': 'Noto Sans',
        'maori': 'Noto Sans',
        'marathi': 'Noto Sans Marathi',
        'mongolian': 'Noto Sans Mongolian',
        'myanmar (burmese)': 'Noto Sans Myanmar',
        'nepali': 'Noto Sans Devanagari',
        'norwegian': 'Noto Sans',
        'odia': 'Noto Sans Oriya',
        'pashto': 'Noto Sans',
        'persian': 'Noto Sans Persian',
        'polish': 'Noto Sans',
        'portuguese': 'Noto Sans',
        'punjabi': 'Noto Sans Gurmukhi',
        'romanian': 'Noto Sans',
        'russian': 'Noto Sans',
        'samoan': 'Noto Sans',
        'scots gaelic': 'Noto Sans',
        'serbian': 'Noto Sans',
        'sesotho': 'Noto Sans',
        'shona': 'Noto Sans',
        'sindhi': 'Noto Sans',
        'sinhala': 'Noto Sans Sinhala',
        'slovak': 'Noto Sans',
        'slovenian': 'Noto Sans',
        'somali': 'Noto Sans',
        'spanish': 'Noto Sans',
        'sundanese': 'Noto Sans',
        'swahili': 'Noto Sans',
        'swedish': 'Noto Sans',
        'tajik': 'Noto Sans',
        'tamil': 'Noto Sans Tamil',
        'telugu': 'Noto Sans Telugu',
        'thai': 'Noto Sans Thai',
        'turkish': 'Noto Sans',
        'ukrainian': 'Noto Sans',
        'urdu': 'Noto Sans Urdu',
        'uyghur': 'Noto Sans',
        'uzbek': 'Noto Sans',
        'vietnamese': 'Noto Sans',
        'welsh': 'Noto Sans',
        'xhosa': 'Noto Sans',
        'yiddish': 'Noto Sans',
        'yoruba': 'Noto Sans',
        'zulu': 'Noto Sans'
    }
    return font_mapping.get(language.lower(), 'Noto Sans')  # Default to Noto Sans



selected_lang_tar = st.selectbox("Select the target language for translation", ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'])

# Button to trigger translation
if st.button("Transcribe and Translate Audio"):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_path = "temp_audio_file"

        # Load the audio using pydub
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Ensure mono channel
        audio = audio.set_frame_rate(16000)  # Ensure frame rate is 16000 Hz

        # Split the audio into chunks (10 sec per chunk)
        chunk_duration_ms = 10000  
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

        # Variables to store full transcription and translation
        full_transcription = ""
        full_translation = ""

        # # --------------------without chunk starts--------------------------------------------------
        # filename = f"chunk.wav"
        # audio.export(filename, format="wav")
        # with open(filename, "rb") as file:
        #     transcription = client.audio.transcriptions.create(
        #         file=(filename, file.read()),  # Required audio file
        #         model="whisper-large-v3",  # Required model for transcription
        #         prompt="transcribe",
        #         response_format="json",  # Optional
        #         temperature=0.0  # Optional
        #     )
        # # Append the chunk transcription to full transcription
        # transcription_text = transcription.text
        # full_transcription += transcription_text + " "
        # # --------------------without chunk ends--------------------------------------------------

        #----------------------------------chunk wise end----------------------------------------------------------

        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Save the chunk to a temporary file
            chunk_filename = f"chunk_{i}.wav"
            chunk.export(chunk_filename, format="wav")

            # Transcribe the chunk using Groq API
            with open(chunk_filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_filename, file.read()),  # Required audio file
                    model="whisper-large-v3",  # Required model for transcription
                    prompt="Transcribe",
                    response_format="json",  # Optional
                    temperature=0.0  # Optional
                )
            # Append the chunk transcription to full transcription
            chunk_transcription_text = transcription.text
            full_transcription += chunk_transcription_text + " "

            # chunk_translation = lt.translate(transcription.text, source=selected_lang_src, target=selected_lang_tar)
            chunk_translation = translate_text(chunk_transcription_text, selected_lang_tar)
            full_translation += chunk_translation + " "

            # Show progress on the frontend
            st.write(f"Processed chunk {i+1}/{len(chunks)}")
            st.audio(chunk_filename, format="wav") 
            st.write(f"Chunk Transcription: {chunk_transcription_text}")
            st.write(f"Chunk Translation: {chunk_translation}")

        #----------------------------------chunk wise end----------------------------------------------------------

        # Show the final combined transcription and translation
        st.write("Final Transcription:")
        st.write(full_transcription)

        st.write(f"Final Translatation:")
        st.write(full_translation)

    elif mic_audio is not None:
        audio_file_like.seek(0)
        buffer_data = audio_file_like.read()
        # Save the uploaded file to a temporary directory
        with open("temp_audio_file", "wb") as f:
            f.write(buffer_data)
        audio_path = "temp_audio_file"

        # Load the audio using pydub
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Ensure mono channel
        audio = audio.set_frame_rate(16000)  # Ensure frame rate is 16000 Hz

        # Split the audio into chunks (10 sec per chunk)
        chunk_duration_ms = 10000  
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

        # Variables to store full transcription and translation
        full_transcription = ""
        full_translation = ""

        # # --------------------without chunk starts--------------------------------------------------
        # filename = f"chunk.wav"
        # audio.export(filename, format="wav")
        # with open(filename, "rb") as file:
        #     transcription = client.audio.transcriptions.create(
        #         file=(filename, file.read()),  # Required audio file
        #         model="whisper-large-v3",  # Required model for transcription
        #         prompt="transcribe",
        #         response_format="json",  # Optional
        #         temperature=0.0  # Optional
        #     )
        # # Append the chunk transcription to full transcription
        # transcription_text = transcription.text
        # full_transcription += transcription_text + " "
        # # --------------------without chunk ends--------------------------------------------------

        #----------------------------------chunk wise end----------------------------------------------------------

        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Save the chunk to a temporary file
            chunk_filename = f"chunk_{i}.wav"
            chunk.export(chunk_filename, format="wav")

            # Transcribe the chunk using Groq API
            with open(chunk_filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_filename, file.read()),  # Required audio file
                    model="whisper-large-v3",  # Required model for transcription
                    prompt="Transcribe",
                    response_format="json",  # Optional
                    temperature=0.0  # Optional
                )
            # Append the chunk transcription to full transcription
            chunk_transcription_text = transcription.text
            full_transcription += chunk_transcription_text + " "

            # chunk_translation = lt.translate(transcription.text, source=selected_lang_src, target=selected_lang_tar)
            chunk_translation = translate_text(chunk_transcription_text, selected_lang_tar)
            full_translation += chunk_translation + " "

            # Show progress on the frontend
            st.write(f"Processed chunk {i+1}/{len(chunks)}")
            st.audio(chunk_filename, format="wav") 
            st.write(f"Chunk Transcription: {chunk_transcription_text}")
            st.write(f"Chunk Translation: {chunk_translation}")

        #----------------------------------chunk wise end----------------------------------------------------------

        # Show the final combined transcription and translation
        st.write("Final Transcription:")
        st.write(full_transcription)

        st.write(f"Final Translation:")
        st.write(full_translation)

    elif uploaded_vedio_file is not None:
        # st.write("Vedio transcription starts...")
        audio_file = video2mp3(vedio_file_name)
        audio_buffer = get_audio_buffer(audio_file)
        # st.audio(audio_buffer)
        # Load the audio using pydub
        audio = AudioSegment.from_file(audio_buffer)
        audio = audio.set_channels(1)  # Ensure mono channel
        audio = audio.set_frame_rate(16000)  # Ensure frame rate is 16000 Hz

        # Split the audio into chunks (3 sec per chunk)
        chunk_duration_ms = 3000  
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

        # Variables to store full transcription and translation
        full_transcription = ""
        full_translation = ""

        # --------------------without chunk starts--------------------------------------------------
        filename = f"chunk.wav"
        audio.export(filename, format="wav")
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Required audio file
                model="whisper-large-v3",  # Required model for transcription
                prompt="transcribe",
                response_format="verbose_json",  # Optional
                temperature=0.0  # Optional
            )
        # Append the chunk transcription to full transcription
        transcription_segment=transcription.segments
        translation_segment=copy.deepcopy(transcription_segment)
        # translation_segment[0]['text']=translate_text(translation_segment[0]['text'], selected_lang_tar)
        
        for seg in translation_segment:
            # st.write(seg['text'])
            seg['text']=translate_text(seg['text'], selected_lang_tar)
            # seg['text']=translate_text(seg['text'], 'english')

        # for i in range(len(transcription_segment)):
        #     st.write(transcription_segment[i]['text'])
        #     st.write(translation_segment[i]['text'])

        # transcription_text = transcription.text
        # full_transcription += transcription_text + " "
        # --------------------without chunk ends--------------------------------------------------

        # #----------------------------------chunk wise end----------------------------------------------------------

        # # Process each chunk
        # for i, chunk in enumerate(chunks):
        #     # Save the chunk to a temporary file
        #     chunk_filename = f"chunk_{i}.wav"
        #     chunk.export(chunk_filename, format="wav")

        #     # Transcribe the chunk using Groq API
        #     with open(chunk_filename, "rb") as file:
        #         transcription = client.audio.transcriptions.create(
        #             file=(chunk_filename, file.read()),  # Required audio file
        #             model="whisper-large-v3",  # Required model for transcription
        #             prompt="Transcribe",
        #             response_format="verbose_json",  # Optional
        #             temperature=0.0  # Optional
        #         )
        #     # Append the chunk transcription to full transcription
        #     chunk_transcription_text = transcription.text
        #     full_transcription += chunk_transcription_text + " "

        #     # chunk_translation = lt.translate(transcription.text, source=selected_lang_src, target=selected_lang_tar)
        #     chunk_translation = translate_text(chunk_transcription_text, selected_lang_tar)
        #     full_translation += chunk_translation + " "

        #     # Show progress on the frontend
        #     st.write(f"Processed chunk {i+1}/{len(chunks)}")
        #     st.audio(chunk_filename, format="wav") 
        #     st.write(f"Chunk Transcription: {chunk_transcription_text}")
        #     st.write(f"Chunk Translation: {chunk_translation}")

        # #----------------------------------chunk wise end----------------------------------------------------------

        # # Show the final combined transcription and translation
        # st.write("Final Transcription:")
        # st.write(full_transcription)

        # st.write(f"Final Translation:")
        # st.write(full_translation)

        # --------------------------subtitle start----------------------------------------------
        subtitle_file = "output_subtitle.vtt"
        write_vtt(translation_segment, subtitle_file)

        output_video = "output_video_with_subtitles.mp4"
        add_subtitles_to_video(vedio_file_name, subtitle_file, output_video, get_font_for_language(selected_lang_tar))
        # add_subtitles_to_video(vedio_file_name, subtitle_file, output_video, 'Noto Sans Devanagari')
        # add_subtitles_to_video(vedio_file_name, subtitle_file, output_video)

        st.video(output_video)
        # --------------------------subtitle end----------------------------------------------
        # #------------------------------------vedio generator--------------------------------------

        # write_vtt(transcription_segment, os.path.join("/", vedio_file_name + ".vtt"))
        # os.system(f'ffmpeg -i "{vedio_file_name}" -vf subtitles="{vedio_file_name}.vtt" "{vedio_file_name}_subtitled.mp4" ')

        # st.video(f"{vedio_file_name}_subtitled.mp4")

    else:
        st.error("Please upload an audio file.")
