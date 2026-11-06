"""
Generador de shorts de historia.

Uso:
    python src/generar_short.py guiones/rey-semlas.yaml
    python src/generar_short.py guiones/rey-semlas.yaml --sin-conexion   (prueba sin IA)

Pasos:
  1. Lee el guion (YAML) y la configuración (config.yaml).
  2. Genera una imagen por escena con un modelo gratuito de Hugging Face.
  3. Genera la voz en off de cada escena (edge-tts) con los tiempos de cada palabra.
  4. Monta cada escena: imagen con zoom lento + subtítulos palabra a palabra.
  5. Une las escenas, añade voz y música, y guarda el vídeo en salida/.
"""
import argparse
import asyncio
import glob
import math
import os
import random
import shutil
import subprocess
import sys
import wave

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1080, 1920, 30
PAUSA = 0.25  # segundos de respiro al final de cada escena


def log(msg):
    print(f"▶ {msg}", flush=True)


# ---------------------------------------------------------------- utilidades
def fuente(tam):
    for ruta in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 os.path.join(RAIZ, "fuentes", "fuente.ttf")):
        if os.path.exists(ruta):
            return ImageFont.truetype(ruta, tam)
    return ImageFont.load_default(tam)


def duracion(ruta):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", ruta], capture_output=True, text=True)
    return float(out.stdout.strip())


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


# ---------------------------------------------------------------- imágenes
def imagen_ia(prompt, destino, cfg):
    """Pide la imagen a un Space gratuito de Hugging Face (por defecto FLUX.1-schnell)."""
    from gradio_client import Client
    ci = cfg["imagenes"]
    token = os.environ.get("HF_TOKEN") or None
    try:
        cliente = Client(ci["space"], token=token, verbose=False)
    except TypeError:  # versiones antiguas de gradio_client
        cliente = Client(ci["space"], hf_token=token, verbose=False)
    res = cliente.predict(prompt=prompt, seed=0, randomize_seed=True,
                          width=ci["ancho"], height=ci["alto"],
                          num_inference_steps=ci["pasos"], api_name=ci["api_name"])
    ruta = res[0] if isinstance(res, (list, tuple)) else res
    if isinstance(ruta, dict):
        ruta = ruta.get("path") or ruta.get("url")
    Image.open(ruta).convert("RGB").save(destino, quality=95)


def imagen_provisional(texto, n, destino):
    """Imagen de relleno si la IA falla (así el vídeo sale igualmente)."""
    colores = [((120, 10, 30), (20, 0, 10)), ((25, 30, 70), (5, 5, 15)),
               ((90, 50, 20), (15, 5, 0)), ((20, 70, 60), (0, 10, 10))]
    a, b = colores[n % len(colores)]
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(x + (z - x) * t) for x, z in zip(a, b)))
    f = fuente(40)
    lineas, linea = [], ""
    for p in texto.split():
        if len(linea + p) > 38:
            lineas.append(linea); linea = ""
        linea += p + " "
    lineas.append(linea)
    for i, l in enumerate(lineas[:8]):
        d.text((80, 300 + i * 60), l, font=f, fill=(255, 255, 255, 180))
    im.save(destino, quality=95)


def preparar_imagenes(guion, cfg, tmp, sin_conexion):
    estilo = guion.get("estilo") or cfg["imagenes"]["estilo"]
    rutas = []
    for i, esc in enumerate(guion["escenas"]):
        destino = os.path.join(tmp, f"img{i}.jpg")
        propia = esc.get("imagen_archivo")
        if propia:  # imagen puesta a mano en el repositorio
            Image.open(os.path.join(RAIZ, propia)).convert("RGB").save(destino, quality=95)
        elif sin_conexion:
            imagen_provisional(esc["imagen"], i, destino)
        else:
            prompt = f"{esc['imagen']}, {estilo}"
            for intento in range(3):
                try:
                    log(f"Imagen {i + 1}/{len(guion['escenas'])} (intento {intento + 1})")
                    imagen_ia(prompt, destino, cfg)
                    break
                except Exception as e:  # cola llena, cuota agotada, Space caído...
                    log(f"  fallo: {str(e)[:150]}")
            else:
                log("  uso imagen provisional")
                imagen_provisional(esc["imagen"], i, destino)
        rutas.append(destino)
    return rutas


# ---------------------------------------------------------------- voz
async def _tts(texto, voz, velocidad, destino):
    import edge_tts
    try:
        com = edge_tts.Communicate(texto, voz, rate=velocidad, boundary="WordBoundary")
    except TypeError:
        com = edge_tts.Communicate(texto, voz, rate=velocidad)
    palabras = []
    with open(destino, "wb") as f:
        async for trozo in com.stream():
            if trozo["type"] == "audio":
                f.write(trozo["data"])
            elif trozo["type"] == "WordBoundary":
                ini = trozo["offset"] / 1e7
                palabras.append([trozo["text"], ini, ini + trozo["duration"] / 1e7])
    return palabras


def silencio(destino, segundos):
    with wave.open(destino, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(b"\x00\x00" * int(44100 * segundos))


def preparar_voz(guion, cfg, tmp, sin_conexion):
    """Devuelve por escena: (ruta_wav, duración, [[palabra, inicio, fin], ...])."""
    cv = cfg["voz"]
    escenas = []
    for i, esc in enumerate(guion["escenas"]):
        texto = esc["texto"].strip()
        wav = os.path.join(tmp, f"voz{i}.wav")
        palabras = []
        ok = False
        if not sin_conexion:
            mp3 = os.path.join(tmp, f"voz{i}.mp3")
            try:
                log(f"Voz {i + 1}/{len(guion['escenas'])}")
                palabras = asyncio.run(_tts(texto, cv["voz"], cv["velocidad"], mp3))
                ffmpeg("-i", mp3, "-af", f"apad=pad_dur={PAUSA}", "-ar", "44100", "-ac", "1", wav)
                ok = True
            except Exception as e:
                log(f"  fallo voz: {str(e)[:150]}")
        if not ok:  # sin voz: duración estimada por número de palabras
            silencio(wav, max(2.0, len(texto.split()) / 2.7) + PAUSA)
        dur = duracion(wav)
        if not palabras:  # reparte las palabras de forma uniforme
            ps = texto.split()
            paso = (dur - PAUSA) / len(ps)
            palabras = [[p, k * paso, (k + 1) * paso] for k, p in enumerate(ps)]
        escenas.append((wav, dur, palabras))
    return escenas


# ---------------------------------------------------------------- subtítulos
def normal(p):
    return "".join(c for c in p.lower() if c.isalnum())


def grupos(palabras, max_chars=14):
    """Agrupa palabras en bloques cortos (estilo TikTok)."""
    gs, cur = [], []
    for p in palabras:
        if cur and (len(" ".join(x[0] for x in cur + [p])) > max_chars or len(cur) >= 3):
            gs.append(cur); cur = []
        cur.append(p)
    if cur:
        gs.append(cur)
    return gs


def dibujar_subtitulo(im, bloques, resaltar, t, cfg):
    cs = cfg["subtitulos"]
    f = fuente(cs["tam"])
    d = ImageDraw.Draw(im)
    actual = None
    for g in bloques:
        if g[0][1] <= t:
            actual = g
    if actual is None:
        return
    textos = [p[0].upper() if cs["mayusculas"] else p[0] for p in actual]
    total = d.textlength(" ".join(textos), font=f)
    x = (W - total) / 2
    y = int(H * cs["altura"])
    for (p, ini, fin), txt in zip(actual, textos):
        activa = ini <= t < fin + 0.05
        clave = normal(p) in resaltar
        color = tuple(cs["color_activa"]) if activa else (tuple(cs["color_clave"]) if clave else (255, 255, 255))
        salto = int(14 * math.exp(-max(0, t - ini) * 12)) if t >= ini else 0
        d.text((x, y - salto), txt, font=f, fill=color, stroke_width=cs["borde"], stroke_fill=(0, 0, 0))
        x += d.textlength(txt + " ", font=f)


# ---------------------------------------------------------------- montaje
def cubrir(img, escala):
    """Ajusta la imagen para cubrir 1080x1920*escala sin deformarla."""
    tw, th = int(W * escala), int(H * escala)
    r = max(tw / img.width, th / img.height)
    img = img.resize((math.ceil(img.width * r), math.ceil(img.height * r)), Image.LANCZOS)
    x, y = (img.width - tw) // 2, (img.height - th) // 2
    return img.crop((x, y, x + tw, y + th))


def render_escena(i, img_ruta, dur, palabras, resaltar, cfg, tmp):
    Z = 1.18
    base = cubrir(Image.open(img_ruta).convert("RGB"), Z)
    # alterna movimiento: zoom in / zoom out y pequeño desplazamiento lateral
    z0, z1 = (1.0, 1.12) if i % 2 == 0 else (1.12, 1.0)
    dx = (-1) ** (i // 2) * 0.04
    bloques = grupos(palabras)
    tmpf, final = os.path.join(tmp, f"esc{i}.tmp.mp4"), os.path.join(tmp, f"esc{i}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                          "-preset", "veryfast", "-pix_fmt", "yuv420p", tmpf], stdin=subprocess.PIPE)
    n = int(round(dur * FPS))
    for k in range(n):
        t = k / FPS
        a = t / dur
        a = a * a * (3 - 2 * a)  # movimiento suave
        z = z0 + (z1 - z0) * a
        w, h = min(base.width, W * Z / z), min(base.height, H * Z / z)
        cx = base.width / 2 + dx * W * (a - .5)
        cy = base.height / 2
        x0 = max(0, min(base.width - w, cx - w / 2))
        y0 = max(0, min(base.height - h, cy - h / 2))
        fr = base.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + w, y0 + h))
        dibujar_subtitulo(fr, bloques, resaltar, t, cfg)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit(f"Error montando la escena {i}")
    os.replace(tmpf, final)
    return final


def elegir_musica(guion):
    if guion.get("musica"):
        return os.path.join(RAIZ, guion["musica"])
    pistas = [f for ext in ("mp3", "wav", "m4a", "ogg") for f in glob.glob(os.path.join(RAIZ, "musica", f"*.{ext}"))]
    return random.choice(pistas) if pistas else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("guion")
    ap.add_argument("--sin-conexion", action="store_true", help="prueba sin IA ni voz")
    args = ap.parse_args()

    with open(os.path.join(RAIZ, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.guion, encoding="utf-8") as f:
        guion = yaml.safe_load(f)

    nombre = os.path.splitext(os.path.basename(args.guion))[0]
    tmp = os.path.join(RAIZ, "tmp", nombre)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    os.makedirs(os.path.join(RAIZ, "salida"), exist_ok=True)

    imagenes = preparar_imagenes(guion, cfg, tmp, args.sin_conexion)
    voces = preparar_voz(guion, cfg, tmp, args.sin_conexion)
    resaltar = {normal(p) for p in guion.get("resaltar", [])}

    escenas = []
    for i, (img, (wav, dur, palabras)) in enumerate(zip(imagenes, voces)):
        log(f"Montando escena {i + 1}/{len(imagenes)} ({dur:.1f} s)")
        escenas.append(render_escena(i, img, dur, palabras, resaltar, cfg, tmp))

    lista = os.path.join(tmp, "lista.txt")
    with open(lista, "w") as f:
        f.writelines(f"file '{os.path.basename(e)}'\n" for e in escenas)
    with open(os.path.join(tmp, "voces.txt"), "w") as f:
        f.writelines(f"file '{os.path.basename(v[0])}'\n" for v in voces)
    video = os.path.join(tmp, "video.mp4")
    voz = os.path.join(tmp, "voz.wav")
    ffmpeg("-f", "concat", "-safe", "0", "-i", lista, "-c", "copy", video)
    ffmpeg("-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "voces.txt"), "-c", "copy", voz)

    salida = os.path.join(RAIZ, "salida", f"{nombre}.mp4")
    musica = elegir_musica(guion)
    vol = cfg["musica"]["volumen"]
    if musica:
        log(f"Música: {os.path.basename(musica)}")
        ffmpeg("-i", video, "-i", voz, "-stream_loop", "-1", "-i", musica,
               "-filter_complex",
               f"[2:a]volume={vol},afade=t=out:st={max(0, duracion(video) - 1.5)}:d=1.5[m];"
               f"[1:a][m]amix=inputs=2:duration=first:normalize=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
               "-shortest", salida + ".tmp.mp4")
    else:
        ffmpeg("-i", video, "-i", voz, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
               "-c:a", "aac", "-b:a", "160k", "-shortest", salida + ".tmp.mp4")
    os.replace(salida + ".tmp.mp4", salida)
    log(f"Listo: {salida} ({duracion(salida):.1f} s)")


if __name__ == "__main__":
    main()
