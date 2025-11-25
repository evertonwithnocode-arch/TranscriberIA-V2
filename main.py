from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid, threading, time, traceback, io
from openai import OpenAI
from pytubefix import YouTube  # 🔥 Mudança: pytubefix em vez de pytube (corrige HTTP 400)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = "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"
client = OpenAI(api_key=openai_api_key)

jobs = {}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def chunk_text(text, chunk_size=2000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])


def process_video(job_id, url):
    try:
        log(f"Job {job_id}: Iniciando processamento da URL: {url}")

        # 🔥 PASSO 1: BAIXAR ÁUDIO COM PYTUBEFIX EM MEMÓRIA
        log(f"Job {job_id}: Baixando áudio com pytubefix...")
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)  # Desabilita OAuth pra evitar prompts desnecessários
        title = yt.title  # Agora funciona sem erro 400
        stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()  # Melhor qualidade de áudio
        if not stream:
            raise ValueError("Nenhum stream de áudio disponível (vídeo privado ou erro)")
        
        buffer = io.BytesIO()  # Buffer em memória
        stream.stream_to_buffer(buffer)  # Baixa direto para bytes
        buffer.seek(0)

        buffer_size = len(buffer.read())  # Log do tamanho pra debug
        log(f"Job {job_id}: Áudio baixado ({buffer_size} bytes) - Título: {title}")

        if buffer_size > 25 * 1024 * 1024:  # Limite OpenAI ~25MB
            raise ValueError("Áudio muito grande (>25MB) - Use AssemblyAI para vídeos longos")

        # 🔥 PASSO 2: TRANSCRIÇÃO COM OPENAI (Whisper)
        buffer.seek(0)  # Reset buffer
        log(f"Job {job_id}: Enviando áudio para transcrição...")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",  # Oficial e estável
            file=("audio.webm", buffer, "audio/webm"),  # MIME type flexível (aceita webm/m4a/mp3)
            language="pt"  # Português BR
        )
        transcribed_text = transcription.text
        log(f"Job {job_id}: Transcrição concluída ({len(transcribed_text)} chars)")

        # 🔥 PASSO 3: RESUMO EM PARTES
        partial_summaries = []
        for idx, chunk in enumerate(chunk_text(transcribed_text)):
            log(f"Job {job_id}: Resumindo chunk {idx+1}")

            summary = client.chat.completions.create(
                model="gpt-4o",  # Modelo válido
                messages=[
                    {
                        "role": "system",
                       "content": (
                            "Você é um assistente especializado em resumir transcrições de reuniões de câmaras de vereadores. "
                            "Gere um resumo claro e objetivo, destacando apenas:\n"
                            "- Principais pontos discutidos\n"
                            "- Decisões tomadas\n"
                            "- Ações definidas\n\n"
                            "Use bullet points, evite repetições e conversas paralelas. "
                            "Comece o resumo sempre com o título: Principais Pontos:"
                        )
                    },
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3
            )

            partial_summaries.append(summary.choices[0].message.content)

        # 🔥 PASSO 4: COMBINA RESUMOS
        combined_prompt = "\n\n".join(partial_summaries)
        log(f"Job {job_id}: Combinando resumos...")

        final_summary = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Combine todos os resumos parciais em um único resumo final claro, organizado em bullet points e começando obrigatoriamente com 'Principais Pontos:'"},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.3
        ).choices[0].message.content

        # 🔥 FINALIZA O JOB
        jobs[job_id] = {
            "status": "done",
            "title": title,
            "transcription": transcribed_text,
            "summary": final_summary,
        }

        log(f"Job {job_id}: Finalizado com sucesso!")

    except Exception as e:
        log(f"Job {job_id}: Erro: {e}")
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