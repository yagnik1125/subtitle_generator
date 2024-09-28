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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
import av
import numpy as np
from twilio.rest import Client
from pytube import YouTube
import yt_dlp
import re
import traceback

# https://pypi.org/project/streamlit-webrtc/
# https://github.com/whitphx/streamlit-stt-app/tree/main
# https://whitphx-streamlit-stt-app-app-deepspeech-m6tt1k.streamlit.app/
# https://console.twilio.com/
# mu6WZ4JWsPctRztBUV16nApe2NQ6AwXH   --->secret key
# account_sid = os.environ["ACe79864f19c78b628906f478fee8cce5e"]
# auth_token = os.environ["0387f33bc7ea64e613896df0ba3b39ba"]

# Initialize the Groq client
client = Groq(api_key="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")

# # Define a class to handle video streaming
# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.frames = []
    
#     def transform(self, frame):
#         self.frames.append(frame)
#         return frame

# Streamlit frontend for audio input and translation
st.title("Subtitle Generator App")

def yt_dlp_download(yt_url:str, output_path:str = None) -> str:
    """
    Downloads the audio track from a specified YouTube video URL using the yt-dlp library, then converts it to an MP3 format file.
    This function configures yt-dlp to extract the best quality audio available and uses FFmpeg (via yt-dlp's postprocessors) to convert the audio to MP3 format. The resulting MP3 file is saved to the specified or default output directory with a filename derived from the video title.
    Args:
        yt_url (str): The URL of the YouTube video from which audio will be downloaded. This should be a valid YouTube video URL.
    Returns:
        str: The absolute file path of the downloaded and converted MP3 file. This path includes the filename which is derived from the original video title.
    Raises:
        yt_dlp.utils.DownloadError: If there is an issue with downloading the video's audio due to reasons such as video unavailability or restrictions.
        
        Exception: For handling unexpected errors during the download and conversion process.
    """
    if output_path is None:
        output_path = os.getcwd()

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(yt_url, download=True)
            file_name = ydl.prepare_filename(result)
            mp3_file_path = file_name.rsplit('.', 1)[0] + '.mp3'
            st.info(f"yt_dlp_download saved YouTube video to file path: {mp3_file_path}")
            return mp3_file_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"yt_dlp_download failed to download audio from URL {yt_url}: {e}")
        # raise
    except Exception as e:
        st.error(f"An unexpected error occurred with yt_dlp_download: {e}")
        st.error(traceback.format_exc())
        # raise

def create_audio_chunks(audio_file: str, chunk_size: int, temp_dir: str) -> List[str]:
    """
    Splits an audio file into smaller segments or chunks based on a specified duration. This function is useful for processing large audio files incrementally or in parallel, which can be beneficial for tasks such as audio analysis or transcription where handling smaller segments might be more manageable.
    AudioSegment can slice an audio file by specifying the start and end times in milliseconds. This allows you to extract precise segments of the audio without needing to process the entire file at once. For example, `audio[1000:2000]` extracts a segment from the 1-second mark to the 2-second mark of the audio file.
    Args:
        audio_file (str): The absolute or relative path to the audio file that needs to be chunked. This file should be accessible and readable.
        
        chunk_size (int): The length of each audio chunk expressed in milliseconds. This value determines how the audio file will be divided. For example, a `chunk_size` of 1000 milliseconds will split the audio into chunks of 1 second each.
        
        temp_dir (str): The directory where the temporary audio chunk files will be stored. This directory will be used to save the output chunk files, and it must have write permissions. If the directory does not exist, it will be created.
    Returns:
        List[str]: A list containing the file paths of all the audio chunks created. Each path in the list represents a single chunk file stored in the specified `temp_dir`. The files are named sequentially based on their order in the original audio file.
    Raises:
        FileNotFoundError: If the `audio_file` does not exist or is inaccessible.
        
        PermissionError: If the script lacks the necessary permissions to read the `audio_file` or write to the `temp_dir`.
        ValueError: If `chunk_size` is set to a non-positive value.
    """
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        st.error(f"create_audio_chunks failed to load audio file {audio_file}: {e}")
        st.error(traceback.format_exc())
        return []

    start = 0
    end = chunk_size
    counter = 0
    chunk_files = []



    while start < len(audio):
        chunk = audio[start:end]
        chunk_file_path = os.path.join(temp_dir, f"{counter}_{file_name}.mp3")
        try:
            chunk.export(chunk_file_path, format="mp3") # Using .mp3 because it's cheaper
            chunk_files.append(chunk_file_path)
        except Exception as e:
            error_message = f"create_audio_chunks failed to export chunk {counter}: {e}"
            st.error(error_message)
            st.error(traceback.format_exc())
            # raise error_message
        start += chunk_size
        end += chunk_size
        counter += 1
    return chunk_files



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
        'afrikaans': 'Roboto',
        'albanian': 'Roboto',
        'amharic': 'Noto Sans Ethiopic',
        'arabic': 'Amiri',
        'armenian': 'Noto Sans Armenian',
        'azerbaijani': 'Roboto',
        'basque': 'Roboto',
        'belarusian': 'Roboto',
        'bengali': 'Noto Sans Bengali',
        'bosnian': 'Roboto',
        'bulgarian': 'Roboto',
        'catalan': 'Roboto',
        'cebuano': 'Roboto',
        'chichewa': 'Roboto',
        'chinese (simplified)': 'Noto Sans CJK SC',
        'chinese (traditional)': 'Noto Sans CJK TC',
        'corsican': 'Roboto',
        'croatian': 'Roboto',
        'czech': 'Roboto',
        'danish': 'Roboto',
        'dutch': 'Roboto',
        'english': 'Roboto',
        'esperanto': 'Roboto',
        'estonian': 'Roboto',
        'filipino': 'Roboto',
        'finnish': 'Roboto',
        'french': 'Roboto',
        'frisian': 'Roboto',
        'galician': 'Roboto',
        'georgian': 'Noto Sans Georgian',
        'german': 'Roboto',
        'greek': 'Noto Sans Greek',
        'gujarati': 'Shruti',  # Custom font for Gujarati
        'haitian creole': 'Roboto',
        'hausa': 'Roboto',
        'hawaiian': 'Roboto',
        'hebrew': 'Noto Sans Hebrew',
        'hindi': 'Noto Sans Devanagari',
        'hmong': 'Roboto',
        'hungarian': 'Roboto',
        'icelandic': 'Roboto',
        'igbo': 'Roboto',
        'indonesian': 'Roboto',
        'irish': 'Roboto',
        'italian': 'Roboto',
        'japanese': 'Noto Sans Japanese',
        'javanese': 'Roboto',
        'kannada': 'Noto Sans Kannada',
        'kazakh': 'Roboto',
        'khmer': 'Noto Sans Khmer',
        'korean': 'Noto Sans Korean',
        'kurdish (kurmanji)': 'Roboto',
        'kyrgyz': 'Roboto',
        'lao': 'Noto Sans Lao',
        'latin': 'Roboto',
        'latvian': 'Roboto',
        'lithuanian': 'Roboto',
        'luxembourgish': 'Roboto',
        'macedonian': 'Roboto',
        'malagasy': 'Roboto',
        'malay': 'Roboto',
        'malayalam': 'Noto Sans Malayalam',
        'maltese': 'Roboto',
        'maori': 'Roboto',
        'marathi': 'Noto Sans Devanagari',
        'mongolian': 'Roboto',
        'myanmar (burmese)': 'Noto Sans Myanmar',
        'nepali': 'Noto Sans Devanagari',
        'norwegian': 'Roboto',
        'odia': 'Noto Sans Oriya',
        'pashto': 'Roboto',
        'persian': 'Noto Sans Persian',
        'polish': 'Roboto',
        'portuguese': 'Roboto',
        'punjabi': 'Noto Sans Gurmukhi',
        'romanian': 'Roboto',
        'russian': 'Roboto',
        'samoan': 'Roboto',
        'scots gaelic': 'Roboto',
        'serbian': 'Roboto',
        'sesotho': 'Roboto',
        'shona': 'Roboto',
        'sindhi': 'Roboto',
        'sinhala': 'Noto Sans Sinhala',
        'slovak': 'Roboto',
        'slovenian': 'Roboto',
        'somali': 'Roboto',
        'spanish': 'Roboto',
        'sundanese': 'Roboto',
        'swahili': 'Roboto',
        'swedish': 'Roboto',
        'tajik': 'Roboto',
        'tamil': 'Noto Sans Tamil',
        'telugu': 'Noto Sans Telugu',
        'thai': 'Noto Sans Thai',
        'turkish': 'Roboto',
        'ukrainian': 'Roboto',
        'urdu': 'Noto Sans Urdu',
        'uyghur': 'Roboto',
        'uzbek': 'Roboto',
        'vietnamese': 'Noto Sans Vietnamese',
        'welsh': 'Roboto',
        'xhosa': 'Roboto',
        'yiddish': 'Roboto',
        'yoruba': 'Roboto',
        'zulu': 'Roboto'
    }
    
    return font_mapping.get(language.lower(), 'Roboto')  # Default to Roboto if no specific font is found


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
    # st.video(uploaded_vedio_file)
    video_capture.release()
    # os.remove(tfile.name)


# Add an input field for the YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link")

# Download and process YouTube video if the link is provided
youtube_video_file_name=""
if youtube_link:
    youtube_url_pattern = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$'
    if not re.match(youtube_url_pattern, youtube_link):
        st.error(f"Invalid YouTube URL: {youtube_link}")
        # raise ValueError("Invalid YouTube URL provided.")
    else:
        st.write("Link OK")

    
    try:
        youtube_video_file_name = yt_dlp_download(youtube_link)
    except Exception as e:
        st.error(f"generate_youtube_transcript_with_groq failed to download YouTube video from URL {youtube_link}: {e}")
        st.error(traceback.format_exc())

    st.video(youtube_video_file_name)
    # chunk_size=5*60000
    # temp_dir = "temp_chunks"
    # chunk_files=[]
    # try:
    #     chunk_files = create_audio_chunks(file_path, chunk_size, temp_dir)
    # except Exception as e:
    #     error_message = f"generate_youtube_transcript_with_groq failed to create audio chunks from file {file_path}: {e}"
    #     st.error(error_message)
    #     st.error(traceback.format_exc())
    #     # raise error_message

    # transcripts = []
    # translations = []
    # for file_name in chunk_files:
    #     try:
    #         st.info(f"Transcribing {file_name}")
    #         filename = f"chunk.wav"
    #         file_name.export(filename, format="wav")
    #         with open(filename, "rb") as file:
    #             transcription = client.audio.transcriptions.create(
    #                 file=(filename, file.read()),  # Required audio file
    #                 model="whisper-large-v3",  # Required model for transcription
    #                 prompt="transcribe",
    #                 response_format="verbose_json",  # Optional
    #                 temperature=0.0  # Optional
    #             )
    #         # Append the chunk transcription to full transcription
    #         transcription_segment=transcription.segments
    #         translation_segment=copy.deepcopy(transcription_segment)
    #         # transcript = transcribe_with_groq(file_name)
    #         # transcripts.append(transcript)
    #         # translation = translate_with_groq(transcript, "English")
    #         # translations.append(translation)
            
    #     except Exception as e:
    #         error_message = f"generate_youtube_transcript_with_groq failed to transcribe file {file_name}: {e}"
    #         st.error(error_message)
    #         st.error(traceback.format_exc())
    #         # raise error_message
    st.write("Youtube video links: ",youtube_link)


selected_lang_tar = st.selectbox("Select the Target language for Subtitle", ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'])


# Button to trigger translation
if st.button("Generate Subtitle"):
    if uploaded_vedio_file is not None:
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

if st.button("Generate Youtube Vedio Subtitle"):
    if youtube_vedio_file_name:
        st.video(youtube_vedio_file_name)
        # audio_file = video2mp3(youtube_vedio_file_name)
        # audio_buffer = get_audio_buffer(audio_file)
        # audio = AudioSegment.from_file(audio_buffer)
        # audio = audio.set_channels(1)
        # audio = audio.set_frame_rate(16000)

        # chunk_duration_ms = 3000  
        # chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]

        # full_transcription = ""
        # full_translation = ""

        # filename = f"chunk.wav"
        # audio.export(filename, format="wav")
        # with open(filename, "rb") as file:
        #     transcription = client.audio.transcriptions.create(
        #         file=(filename, file.read()),
        #         model="whisper-large-v3",
        #         prompt="transcribe",
        #         response_format="verbose_json",
        #         temperature=0.0
        #     )
        # transcription_segment = transcription.segments
        # translation_segment = copy.deepcopy(transcription_segment)
        
        # for seg in translation_segment:
        #     seg['text'] = translate_text(seg['text'], selected_lang_tar)

        # subtitle_file = "output_subtitle.vtt"
        # write_vtt(translation_segment, subtitle_file)

        # output_video = "output_video_with_subtitles.mp4"
        # add_subtitles_to_video(vedio_file_name, subtitle_file, output_video, get_font_for_language(selected_lang_tar))

        # st.video(output_video)

    else:
        st.error("Please upload a video file or provide a YouTube link.")
