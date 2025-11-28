from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp, uuid, os, threading, time
from openai import OpenAI
from pydub import AudioSegment
import traceback
from fastapi.responses import JSONResponse
from fastapi import Request

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://transcribeia.lovable.app",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://localhost",
    "http://127.0.0.1"
]

@app.middleware("http")
async def block_disallowed_origins(request: Request, call_next):
    origin = request.headers.get("origin")

    # permite requisições sem origin (Postman, curl)
    if origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse(
            {"detail": "Origin not allowed"},
            status_code=403
        )

    return await call_next(request)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # <-- correção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)



openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não encontrada! Configure ela no ambiente.")

client = OpenAI(api_key=openai_api_key)

jobs = {}

# -----------------------------------------
# LOG
# -----------------------------------------
def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# -----------------------------------------
# DIVISÃO DE TEXTO
# -----------------------------------------
def chunk_text(text, chunk_size=2000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])


# -----------------------------------------
# DIVISÃO DO ÁUDIO
# -----------------------------------------
def split_audio(file_path, max_duration_sec=1390):
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


# =====================================================================
# PROCESSAMENTO COMPLETO DO VÍDEO
# =====================================================================
def process_video(job_id, url):

    try:
        cookies_path = "cookies.txt"

        # ----------------------------
        # VERIFICA COOKIES
        # ----------------------------
        if not os.path.isfile(cookies_path):
            raise FileNotFoundError("Arquivo cookies.txt não encontrado em /app/cookies.txt")

        log(f"Job {job_id}: cookies.txt encontrado ({cookies_path})")

        # ----------------------------
        # OBTÉM TÍTULO
        # ----------------------------
        log(f"Job {job_id}: Iniciando download: {url}")

        audio_id = str(uuid.uuid4())
        base_path = os.path.join(OUTPUT_DIR, audio_id)

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get("title", "Vídeo do YouTube")

        log(f"Job {job_id}: Título: {video_title}")

        # ----------------------------
        # BAIXA O ÁUDIO
        # ----------------------------
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": base_path,
            "cookiefile": cookies_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192"
                }
            ]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        final_path = base_path + ".mp3"
        log(f"Job {job_id}: Áudio baixado: {final_path}")

        # ----------------------------
        # DIVIDE O ÁUDIO
        # ----------------------------
        audio_chunks = split_audio(final_path)
        log(f"Job {job_id}: Total partes: {len(audio_chunks)}")

        # ----------------------------
        # TRANSCRIÇÃO
        # ----------------------------
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
        log(f"Job {job_id}: Transcrição concluída ({len(transcribed_text)} chars)")

        # =====================================================================
        # RESUMO (PARA ATA COMPLETA)
        # =====================================================================

        ATA_PROMPT = (
            "Você é um assistente especializado em gerar **atas completas de sessões de câmaras de vereadores**.\n\n"
            "REGRAS ESTRITAS:\n"
            "1. A ata deve ser **detalhada**, nunca superficial.\n"
            "2. Incluir **fala de cada vereador**, com clareza e precisão.\n"
            "3. Para cada projeto, incluir:\n"
            "   - número de votos favoráveis\n"
            "   - votos contrários (se houver)\n"
            "   - abstenções (se houver)\n"
            "   - quem pediu a palavra e o que disse\n"
            "4. Organizar obrigatoriamente em:\n"
            "   **I – Expediente**\n"
            "   **II – Ordem do Dia**\n"
            "   **III – Explicações Pessoais**\n"
            "5. Não inventar nomes ou fatos: usar somente o que estiver no áudio.\n"
            "6. A linguagem deve ser **formal, legislativa e clara**.\n"
            "7. Transformar fal falas soltas em narrativa institucional.\n"
            "8. Nunca encurtar demais: manter riqueza de detalhes.\n\n"
            "Agora, gere a ATA do seguinte trecho:\n\n"
        )

        partial_summaries = []

        for i, chunk in enumerate(chunk_text(transcribed_text)):
            log(f"Job {job_id}: Gerando ata do chunk {i+1}")

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": ATA_PROMPT},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.2
            )

            partial_summaries.append(response.choices[0].message.content)

        # ----------------------------
        # ATA FINAL COMBINADA
        # ----------------------------
        combined = "\n\n".join(partial_summaries)

        final_ata = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você deve combinar vários trechos de ata em uma única ata completa, "
                        "mantendo fidelidade total ao conteúdo e organização legislativa:"
                        "\nI – Expediente\nII – Ordem do Dia\nIII – Explicações Pessoais\n"
                        "Não resuma: apenas una, organize e ajuste a redação."
                    )
                },
                {"role": "user", "content": combined}
            ],
            temperature=0.1
        )

        ata_final = final_ata.choices[0].message.content

        jobs[job_id] = {
            "status": "done",
            "title": video_title,
            "transcription": transcribed_text,
            "summary": ata_final
        }

        log(f"Job {job_id}: Finalizado!")

    except Exception as e:
        log(f"Job {job_id}: ERRO: {str(e)}")
        log(traceback.format_exc())
        jobs[job_id] = {"status": "error", "error": str(e)}



# =====================================================================
# ROTAS FASTAPI
# =====================================================================

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


@app.get("/")
def root():
    return {"status": "ok", "message": "API rodando"}