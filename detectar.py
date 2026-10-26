"""
detectar.py  v2
Detecta motos, bicicletas y perros en un video.
Guarda clips individuales solo cuando hay deteccion,
con buffer configurable de segundos antes y despues.

Uso:
    python detectar.py --video cam.mp4
    python detectar.py --video cam.mp4 --antes 15 --despues 15 --saltar 3
    python detectar.py --video cam.mp4 --modelo yolo11s.pt --confianza 0.3 --mostrar

Requisitos:
    pip install ultralytics opencv-python
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASES_OBJETIVO = {1: "bicycle", 3: "motorcycle", 16: "dog"}
COLORES         = {1: (255, 200, 0), 3: (0, 0, 255), 16: (0, 255, 0)}


def parse_args():
    p = argparse.ArgumentParser(
        description="Detecta motos, bicicletas y perros. Guarda clips con la deteccion."
    )
    p.add_argument("--video",     required=True,          help="Ruta del video a analizar")
    p.add_argument("--modelo",    default="yolo11n.pt",   help="Modelo YOLO (default: yolo11n.pt)")
    p.add_argument("--saltar",    type=int,   default=1,  help="Analizar 1 de cada N frames (default: 1)")
    p.add_argument("--confianza", type=float, default=0.3, help="Umbral de confianza (default: 0.3)")
    p.add_argument("--antes",     type=int,   default=15, help="Segundos antes de la deteccion (default: 15)")
    p.add_argument("--despues",   type=int,   default=15, help="Segundos despues de la deteccion (default: 15)")
    p.add_argument("--mostrar",   action="store_true",    help="Mostrar ventana en tiempo real durante el paso 1")
    p.add_argument("--salida",    default=None,           help="Carpeta de salida para los clips")
    return p.parse_args()


def dibujar(frame, detecciones):
    for cls_id, track_id, conf, x1, y1, x2, y2 in detecciones:
        nombre = CLASES_OBJETIVO.get(cls_id, str(cls_id))
        color  = COLORES.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{nombre} {conf:.2f}"
        if track_id != -1:
            etiqueta += f" ID:{track_id}"
        cv2.putText(frame, etiqueta, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def agrupar_clips(detecciones_por_frame, buf_antes, buf_despues, total_frames):
    """
    Agrupa frames con deteccion en clips (inicio, fin).
    Clips que se solapan quedan fusionados en uno solo.
    """
    if not detecciones_por_frame:
        return []

    frames = sorted(detecciones_por_frame)
    clips  = []
    inicio = max(0, frames[0] - buf_antes)
    fin    = min(total_frames - 1, frames[0] + buf_despues)

    for f in frames[1:]:
        nuevo_inicio = max(0, f - buf_antes)
        nuevo_fin    = min(total_frames - 1, f + buf_despues)

        if nuevo_inicio <= fin:
            fin = max(fin, nuevo_fin)
        else:
            clips.append((inicio, fin))
            inicio = nuevo_inicio
            fin    = nuevo_fin

    clips.append((inicio, fin))
    return clips


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: no se encontro el video '{video_path}'")
        sys.exit(1)

    salida_dir = Path(args.salida) if args.salida else \
                 video_path.parent / (video_path.stem + "_clips")
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo '{args.modelo}' ...")
    modelo = YOLO(args.modelo)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: no se pudo abrir el video")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    ancho        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buf_antes    = int(fps * args.antes)
    buf_despues  = int(fps * args.despues)

    print(f"Video : {ancho}x{alto} @ {fps:.1f} fps | {total_frames} frames")
    print(f"Buffer: {args.antes}s antes / {args.despues}s despues")
    print(f"Clips : {salida_dir}/")

    # -------------------------------------------------------------------
    # PASO 1: recorrer el video y registrar en que frames hay detecciones
    # -------------------------------------------------------------------
    print("\n--- PASO 1: Deteccion ---")

    clases_lista          = list(CLASES_OBJETIVO.keys())
    detecciones_por_frame = {}
    contadores            = defaultdict(set)

    frame_num = 0
    t0        = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % args.saltar != 0:
                continue

            resultados = modelo.track(
                frame,
                classes=clases_lista,
                conf=args.confianza,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            r = resultados[0]
            if r.boxes is not None and len(r.boxes) > 0:
                dets = []
                for box in r.boxes:
                    cls_id       = int(box.cls[0])
                    nombre       = CLASES_OBJETIVO.get(cls_id, str(cls_id))
                    conf         = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    track_id     = int(box.id[0]) if box.id is not None else -1

                    if track_id != -1:
                        contadores[nombre].add(track_id)
                    dets.append((cls_id, track_id, conf, x1, y1, x2, y2))

                detecciones_por_frame[frame_num] = dets

            if args.mostrar:
                vis = frame.copy()
                if frame_num in detecciones_por_frame:
                    vis = dibujar(vis, detecciones_por_frame[frame_num])
                cv2.imshow("Paso 1 - Deteccion (ESC para salir)", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Interrumpido por ESC.")
                    break

            if frame_num % 500 == 0 or frame_num == total_frames:
                pct = frame_num / total_frames * 100 if total_frames else 0
                print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)  |  "
                      f"Frames con deteccion: {len(detecciones_por_frame)}")

    except KeyboardInterrupt:
        print("\nInterrumpido con Ctrl+C.")
    finally:
        cap.release()
        if args.mostrar:
            cv2.destroyAllWindows()

    print(f"\nPaso 1 terminado en {time.time()-t0:.1f}s")
    print(f"Frames analizados    : {frame_num}")
    print(f"Frames con deteccion : {len(detecciones_por_frame)}")
    for nombre, ids in contadores.items():
        print(f"  {nombre}: {len(ids)} objeto(s) unico(s)")

    if not detecciones_por_frame:
        print("\nNo hubo detecciones. No se generan clips.")
        return

    # -------------------------------------------------------------------
    # PASO 2: agrupar en clips y extraerlos del video original
    # -------------------------------------------------------------------
    print(f"\n--- PASO 2: Generando clips ---")

    clips  = agrupar_clips(detecciones_por_frame, buf_antes, buf_despues, total_frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap2   = cv2.VideoCapture(str(video_path))

    print(f"Clips a generar: {len(clips)}")

    for i, (inicio, fin) in enumerate(clips, 1):
        duracion_s  = (fin - inicio) / fps
        nombre_clip = salida_dir / f"clip_{i:03d}_frame{inicio:06d}.mp4"

        print(f"  [{i}/{len(clips)}] frames {inicio}-{fin} "
              f"({duracion_s:.1f}s) -> {nombre_clip.name}")

        writer = cv2.VideoWriter(str(nombre_clip), fourcc, fps, (ancho, alto))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, inicio)

        for f in range(inicio, fin + 1):
            ret, frame = cap2.read()
            if not ret:
                break
            if f in detecciones_por_frame:
                frame = dibujar(frame, detecciones_por_frame[f])
            writer.write(frame)

        writer.release()

    cap2.release()

    print(f"\n--- Resumen ---")
    print(f"Clips guardados: {len(clips)} en '{salida_dir}/'")
    for nombre, ids in contadores.items():
        print(f"  {nombre}: {len(ids)} objeto(s) unico(s)")
    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()