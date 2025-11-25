from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp, uuid, os, threading, time
from openai import OpenAI
import traceback
import subprocess

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

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
client = OpenAI(api_key=openai_api_key)

jobs = {}

def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def chunk_text(text, chunk_size=2000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def split_audio(file_path, max_duration_sec=1390):
    chunks = []
    output_template = f"{file_path}_part%03d.mp3"

    # Rodar o ffmpeg para dividir
    cmd = [
        "ffmpeg", "-i", file_path,
        "-f", "segment",
        "-segment_time", str(max_duration_sec),
        "-c", "copy",
        output_template
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Coletar arquivos gerados
    base_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    for file in sorted(os.listdir(base_dir)):
        if file.startswith(base_name + "_part") and file.endswith(".mp3"):
            chunks.append(os.path.join(base_dir, file))

    return chunks


def process_video(job_id, url):
    try:
        cookies_path = "cookies.txt"

        # 🔥 NOVO → Verificação + EXIBIÇÃO DO cookies.txt
        if not os.path.isfile(cookies_path):
            raise FileNotFoundError("Arquivo cookies.txt não encontrado em /app/cookies.txt")
        else:
            log(f"Job {job_id}: cookies.txt encontrado ({cookies_path})")

            # Exibe tamanho
            file_size = os.path.getsize(cookies_path)
            log(f"Job {job_id}: Tamanho do cookies.txt: {file_size} bytes")

            # Exibe conteúdo
            with open(cookies_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            log(f"Job {job_id}: Conteúdo do cookies.txt:\n\n{content}\n\n=== FIM DO ARQUIVO ===")

        log(f"Job {job_id}: Iniciando download do vídeo: {url}")
        audio_id = str(uuid.uuid4())
        base_path = os.path.join(OUTPUT_DIR, audio_id)

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get("title", "Vídeo do YouTube")
            log(f"Job {job_id}: Título obtido: {video_title}")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": base_path,
            "cookiefile": cookies_path,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        final_path = base_path + ".mp3"
        log(f"Job {job_id}: Áudio baixado: {final_path}")

        audio_chunks = split_audio(final_path)
        log(f"Job {job_id}: Dividido em {len(audio_chunks)} partes")

        transcriptions = []
        for i, chunk_file in enumerate(audio_chunks):
            log(f"Job {job_id}: Transcrevendo parte {i+1}")
            with open(chunk_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file
                )
                transcriptions.append(transcript.text)

        transcribed_text = "\n".join(transcriptions)
        log(f"Job {job_id}: Transcrição finalizada ({len(transcribed_text)} chars)")

        partial_summaries = []
        for i, chunk in enumerate(chunk_text(transcribed_text)):
            log(f"Job {job_id}: Resumindo chunk {i+1}")
            summary_response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
            {
                 "role": "system",
                 "content": (
                 "Você é um assistente especializado em resumir transcrições de reuniões de câmaras de vereadores. "
                 "Seu objetivo é gerar um resumo claro e objetivo destacando apenas os principais pontos discutidos, "
                 "decisões tomadas e ações definidas. "
                 "Evite repetições, comentários irrelevantes ou conversas paralelas. "
                 "O resumo deve ser organizado em tópicos e começar com o título 'Principais Pontos:'"
            )
            },
                {"role": "user", "content": "Resuma:\n\n" + chunk}
             ],
                temperature=0.3
            )
            partial_summaries.append(summary_response.choices[0].message.content)

        combined_prompt = "\n\n".join(partial_summaries)
        final_summary_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Você é um assistente que gera resumos."},
                {"role": "user", "content": "Combine os resumos:\n\n" + combined_prompt}
            ],
            temperature=0.3
        )

        summarized_text = final_summary_response.choices[0].message.content

        jobs[job_id] = {
            "status": "done",
            "title": video_title,
            "transcription": transcribed_text,
            "summary": summarized_text
        }

        log(f"Job {job_id}: Finalizado com sucesso!")

    except Exception as e:
        log(f"Job {job_id}: Erro: {str(e)}")
        log(traceback.format_exc())
        jobs[job_id] = {"status": "error", "error": str(e)}



@app.post("/start-job")
async def start_job(url: str = Form(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    log(f"Job {job_id}: Criado")
    threading.Thread(target=process_video, args=(job_id, url)).start()
    return {"job_id": job_id}


@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return jobs[job_id]
