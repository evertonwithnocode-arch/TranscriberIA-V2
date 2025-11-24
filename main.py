from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp, uuid, os, threading, time
from openai import OpenAI
from pydub import AudioSegment
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
client = OpenAI(api_key=openai_api_key)

# Armazena os jobs em memória
jobs = {}

def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def chunk_text(text, chunk_size=2000):
    """Divide o texto em partes menores para evitar limite de tokens."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def split_audio(file_path, max_duration_sec=1390):
    """Divide o áudio em partes menores que o limite do modelo (1400s)."""
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = max_duration_sec * 1000
    chunks = []
    total_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start+chunk_length_ms]
        chunk_file = f"{file_path}_part{i}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunks.append(chunk_file)
        log(f"Chunk {i+1}/{total_chunks}: {len(chunk)/1000:.2f}s -> {chunk_file}")
    return chunks

def process_video(job_id, url):
    try:
        log(f"Job {job_id}: Iniciando download do vídeo: {url}")
        audio_id = str(uuid.uuid4())
        base_path = os.path.join(OUTPUT_DIR, audio_id)

        # Extrai metadados
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get("title", "Vídeo do YouTube")
            log(f"Job {job_id}: Título do vídeo obtido: {video_title}")

        # Baixa o áudio
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": base_path,
            "cookies": "/app/cookies.txt",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        final_path = base_path + ".mp3"
        log(f"Job {job_id}: Áudio baixado e convertido para MP3: {final_path}")

        # Divide o áudio em partes menores
        audio_chunks = split_audio(final_path)
        log(f"Job {job_id}: Áudio dividido em {len(audio_chunks)} partes para transcrição")

        # Transcrição
        transcriptions = []
        for i, chunk_file in enumerate(audio_chunks):
            log(f"Job {job_id}: Transcrevendo chunk {i+1}")
            with open(chunk_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file
                )
                transcriptions.append(transcript.text)
        transcribed_text = "\n".join(transcriptions)
        log(f"Job {job_id}: Transcrição completa, tamanho: {len(transcribed_text)} caracteres")

        # Resumo em partes
        log(f"Job {job_id}: Iniciando resumo em partes com GPT-4.1")
        partial_summaries = []
        for i, chunk in enumerate(chunk_text(transcribed_text)):
            log(f"Job {job_id}: Resumindo chunk {i+1}")
            summary_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Você é um assistente que gera resumos claros e objetivos."},
                    {"role": "user", "content": (
                        "Resuma o seguinte texto destacando os principais pontos, decisões e ações. "
                        "Comece com 'Principais Pontos:'\n\n" + chunk
                    )}
                ],
                temperature=0.3
            )
            partial_summaries.append(summary_response.choices[0].message.content)

        # Combina os resumos parciais em um resumo final
        log(f"Job {job_id}: Combinando resumos parciais")
        combined_prompt = "Combine os seguintes resumos parciais em um único resumo coeso, sem repetições:\n\n" + "\n\n".join(partial_summaries)
        final_summary_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Você é um assistente que gera resumos claros e objetivos."},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.3
        )
        summarized_text = final_summary_response.choices[0].message.content
        log(f"Job {job_id}: Resumo final gerado, tamanho: {len(summarized_text)} caracteres")

        jobs[job_id] = {
            "status": "done",
            "title": video_title,
            "transcription": transcribed_text,
            "summary": summarized_text
        }
        log(f"Job {job_id}: Job concluído com sucesso")

    except Exception as e:
        log(f"Job {job_id}: Erro encontrado - {str(e)}")
        log(traceback.format_exc())
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.post("/start-job")
async def start_job(url: str = Form(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    log(f"Job {job_id}: Recebido pedido de transcrição")
    threading.Thread(target=process_video, args=(job_id, url)).start()
    return {"job_id": job_id}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job
