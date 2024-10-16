from flask import Flask, render_template, request, jsonify
from pytube import YouTube
from threading import Thread
import os
import string
from datetime import datetime

app = Flask(__name__)

def generate_unique_filename(title, extension):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned_title = ''.join(c for c in title if c in valid_chars)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{cleaned_title}_{timestamp}.{extension}"
    return filename

def download_video(video_url, download_path):
    yt = YouTube(video_url)
    stream = yt.streams.get_highest_resolution()
    filename = generate_unique_filename(yt.title, 'mp4')
    stream.download(output_path=download_path, filename=filename)

def download_audio(video_url, download_path):
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    filename = generate_unique_filename(yt.title, 'mp3')
    stream.download(output_path=download_path, filename=filename)

@app.route('/')
def index():
    return render_template('index.html')