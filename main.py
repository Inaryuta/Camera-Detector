"""
main.py
Detección de motos, bicicletas y perros en un video usando YOLO (Ultralytics).

Uso:
    python detectar.py --video ruta/al/video.mp4
    python detectar.py --video ruta/al/video.mp4 --modelo yolo11s.pt --mostrar
    python detectar.py --video ruta/al/video.mp4 --salida resultado.mp4

Requisitos:
    pip install ultralytics opencv-python
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Clases COCO que nos interesan (índices estándar usados por YOLO/ultralytics):
# 1 = bicycle, 3 = motorcycle, 16 = dog
CLASES_OBJETIVO = {
    1: "bicycle",
    3: "motorcycle",
    16: "dog",
}

# Colores BGR para cada clase (para dibujar los cuadros)
COLORES = {
    1: (255, 200, 0),    # bicicleta - celeste
    3: (0, 0, 255),      # moto - rojo
    16: (0, 255, 0),     # perro - verde
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta motos, bicicletas y perros en un video usando YOLO."
    )
    parser.add_argument("--video", required=True, help="Ruta del video a procesar")
    parser.add_argument(
        "--modelo",
        default="yolo11n.pt",
        help="Modelo YOLO a usar (default: yolo11n.pt, el mas rapido)",
    )
    parser.add_argument(
        "--salida",
        default=None,
        help="Ruta del video de salida con anotaciones (default: <video>_detectado.mp4)",
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Mostrar la ventana de video en tiempo real mientras procesa",
    )
    parser.add_argument(
        "--confianza",
        type=float,
        default=0.4,
        help="Umbral de confianza minimo para mostrar una deteccion (default: 0.4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: no se encontro el video '{video_path}'")
        sys.exit(1)

    salida_path = Path(args.salida) if args.salida else video_path.with_name(
        video_path.stem + "_detectado.mp4"
    )

    print(f"Cargando modelo '{args.modelo}' ...")
    modelo = YOLO(args.modelo)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir el video '{video_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(salida_path), fourcc, fps, (ancho, alto))

    print(f"Video: {ancho}x{alto} @ {fps:.1f} fps, {total_frames} frames")
    print(f"Guardando resultado en: {salida_path}")
    print("Procesando... (Ctrl+C para detener)")

    contador_por_clase = {nombre: 0 for nombre in CLASES_OBJETIVO.values()}
    ids_vistos = {nombre: set() for nombre in CLASES_OBJETIVO.values()}

    frame_num = 0
    t_inicio = time.time()

    clases_lista = list(CLASES_OBJETIVO.keys())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            resultados = modelo.track(
                frame,
                classes=clases_lista,
                conf=args.confianza,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            r = resultados[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    nombre = CLASES_OBJETIVO.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    if track_id != -1 and track_id not in ids_vistos[nombre]:
                        ids_vistos[nombre].add(track_id)
                        contador_por_clase[nombre] += 1

                    color = COLORES.get(cls_id, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    etiqueta = f"{nombre} {conf:.2f}"
                    if track_id != -1:
                        etiqueta += f" ID:{track_id}"
                    cv2.putText(
                        frame, etiqueta, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )

            writer.write(frame)

            if args.mostrar:
                cv2.imshow("Deteccion - motos, bicicletas y perros", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                    print("Interrumpido por el usuario (ESC).")
                    break

            if frame_num % 50 == 0 or frame_num == total_frames:
                pct = (frame_num / total_frames * 100) if total_frames else 0
                print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario (Ctrl+C).")

    finally:
        cap.release()
        writer.release()
        if args.mostrar:
            cv2.destroyAllWindows()

    duracion = time.time() - t_inicio
    print("\n--- Resumen ---")
    print(f"Frames procesados: {frame_num}")
    if duracion > 0:
        print(f"Tiempo total: {duracion:.1f} s ({frame_num / duracion:.1f} fps promedio)")
    for nombre, total in contador_por_clase.items():
        print(f"  {nombre}: {total} objeto(s) unico(s) detectado(s)")
    print(f"Video con anotaciones guardado en: {salida_path}")


if __name__ == "__main__":
    main()