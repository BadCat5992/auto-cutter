import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import os
import json
import time
import psutil
from faster_whisper import WhisperModel
import ollama

# 🔧 Settings
OLLAMA_MODEL = "command-r"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
pipeline_lock = threading.Lock()

def check_ollama_available():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return OLLAMA_MODEL in result.stdout
    except Exception as e:
        print("⚠️ Ollama nicht erreichbar:", e)
        return False

def is_ollama_model_running(model_name="command-r"):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info["cmdline"] and any(model_name in arg for arg in proc.info["cmdline"]):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def kill_ollama_model_processes(model_name="command-r"):
    print("🧹 Beende laufende Ollama-Prozesse...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info["cmdline"] and any(model_name in arg for arg in proc.info["cmdline"]):
                print(f"💀 Kille Prozess {proc.pid} ({' '.join(proc.info['cmdline'])})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def transcribe_audio(video_path):
    print("🎙️ Extrahiere Audio...")
    audio_path = "temp_audio.wav"

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg ist nicht installiert oder nicht im PATH")

    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", audio_path], check=True)
        print("✅ Audio extrahiert")

        print("🧠 Lade Whisper-Modell...")
        model = WhisperModel("medium", device="cpu", compute_type="int8")
        print("✅ Whisper-Modell geladen")

        print("✍️ Transkribiere...")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([seg.text for seg in segments])
        print("📝 Transkript fertig!")
        return transcript

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def extract_cut_ranges(text):
    print("🤖 Frage LLM nach Schnittstellen...")

    prompt = f"""
Gib mir die spannendsten Ausschnitte in diesem JSON-Format zurück:
{{"cuts": [[start1, end1], [start2, end2], ...]}}
Mache bitte so viele cuts wie möglich!
Nur den JSON bitte, kein anderes Gerede!

Transkript:
{text}
"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "/set format json"},
        {"role": "user", "content": prompt}
    ])

    content = response["message"]["content"]
    print("🧠 LLM-Antwort:", content)

    try:
        parsed = json.loads(content)
        ranges = [tuple(sorted([int(start), int(end)])) for start, end in parsed.get("cuts", [])]
        ranges = sorted(ranges)
        print("⏱️ Gefundene Schnittbereiche:", ranges)
        return ranges
    except Exception as e:
        print("⚠️ Fehler beim Parsen der JSON-Antwort:", e)
        return []

def cut_clips_ffmpeg(input_path, ranges, temp_dir="temp_clips"):
    print("✂️ Schneide Clips mit ffmpeg...")
    os.makedirs(temp_dir, exist_ok=True)
    clip_paths = []
    for i, (start, end) in enumerate(ranges):
        out_path = os.path.join(temp_dir, f"clip_{i}.mp4")
        duration = end - start
        print(f"🔪 Schneide Clip {i}: {start}s bis {end}s")
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            out_path
        ]
        subprocess.run(cmd, check=True)
        clip_paths.append(out_path)
    return clip_paths

def concat_clips_ffmpeg(clip_paths, output_path, temp_dir="temp_clips"):
    print("🔗 Füge Clips zusammen...")
    list_file = os.path.join(temp_dir, "clips.txt")
    with open(list_file, "w") as f:
        for clip in clip_paths:
            # Nur Dateiname, weil wir cwd setzen
            f.write(f"file '{os.path.basename(clip)}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "clips.txt", "-c", "copy", output_path
    ]
    subprocess.run(cmd, check=True, cwd=temp_dir)
    print("✅ Video gespeichert:", output_path)

def cut_video(video_path, ranges, output_path):
    clips = cut_clips_ffmpeg(video_path, ranges)
    if clips:
        concat_clips_ffmpeg(clips, output_path)
    else:
        raise RuntimeError("Keine Clips zum Zusammenfügen gefunden!")

def run_pipeline_with_gui(filepath):
    with pipeline_lock:
        try:
            label.config(text="🧠 Prüfe Ollama Erreichbarkeit...")
            if not check_ollama_available():
                label.config(text="❌ Ollama läuft nicht! Bitte installiere es oder prüfe die Verbindung.")
                return

            if not is_ollama_model_running():
                print("🚀 Starte Ollama-Modell...")
                subprocess.Popen(["ollama", "run", OLLAMA_MODEL])
                time.sleep(8)  # 🕐 Warte bis Modell geladen ist

            label.config(text="🎙️ Transkribiere...")
            transcript = transcribe_audio(filepath)

            label.config(text="🧠 Finde Highlights...")
            ranges = extract_cut_ranges(transcript)

            if not ranges:
                label.config(text="❌ Keine Highlights erkannt.")
                return

            label.config(text="✂️ Schneide Video...")
            output_path = "highlight_output.mp4"
            cut_video(filepath, ranges, output_path)

            label.config(text="✅ Fertig! Gespeichert als highlight_output.mp4")

        except Exception as e:
            label.config(text=f"💥 Fehler: {str(e)}")
            print("💥 Fehler:", e)
        finally:
            kill_ollama_model_processes(OLLAMA_MODEL)

def select_file():
    filepath = filedialog.askopenfilename(title="Wähle ein Video")
    if filepath:
        label.config(text="⏳ Starte Verarbeitung...")
        threading.Thread(target=run_pipeline_with_gui, args=(filepath,), daemon=True).start()

# 🖥️ GUI Setup
root = tk.Tk()
root.title("🎮 AutoCut by Julius")
root.geometry("420x180")
label = tk.Label(root, text="Wähle dein Gaming-Video für automatische Highlights!", pady=20, wraplength=400)
label.pack()
button = tk.Button(root, text="📂 Video auswählen", command=select_file, height=2, width=30)
button.pack()
root.mainloop()

