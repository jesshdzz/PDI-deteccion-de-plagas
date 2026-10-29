"""
Clasificación de imágenes por COMPLEJIDAD PARA IDENTIFICAR LA HOJA (versión 2).

Uso:
    python clasificar_complejidad.py RUTA_A_train [imagenes_por_categoria]

La complejidad se mide por lo difícil que resulta aislar la hoja con métodos simples:
  - eta_gris : separabilidad de Otsu en gris (1 = fondo y hoja muy distintos en intensidad)
  - iou_gs   : coincidencia entre Otsu en gris y Otsu en saturación (baja = sombras / fondo irregular)
  - frag     : fracción de la máscara que forma un solo bloque (baja = hoja rota en pedazos)
  - hoja_marco: % de píxeles del marco exterior que quedan como hoja (alto = la hoja toca el borde)
Reglas (umbrales provisionales; ajústalos con las hojas de contacto):
  Tipo 3 (compleja)  : frag < FRAG_MIN  o  hoja_marco > MARCO_MAX  o  eta_sat < ETA_S_MIN
  Tipo 1 (simple)    : iou_gs >= IOU_GS_MIN  y  eta_gris >= ETA_G_MIN
  Tipo 2 (con sombras o fondo irregular): el resto
Genera: complejidad_caracteristicas.csv, complejidad_resumen.csv y hoja_contacto_tipo{1,2,3}.png
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import segmentacion as sg

FRAG_MIN, MARCO_MAX, ETA_S_MIN = 0.80, 10.0, 0.50
IOU_GS_MIN, ETA_G_MIN = 0.60, 0.65
SEMILLA, EXTS = 0, {".jpg", ".jpeg", ".png"}


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def caracteristicas(rgb):
    p = sg.mediana(rgb, 3)
    g = sg.a_gris(p)
    s = sg.a_uint8(sg.a_hsv(p)[..., 1])
    tg, eg = sg.otsu(g)
    ts, es = sg.otsu(s)
    mg = sg.limpiar_mascara(sg.mascara_hoja_por_umbral(g, tg))
    ms = sg.limpiar_mascara(sg.mascara_hoja_por_umbral(s, ts))
    et, _c = sg.segmentar_kmeans(p, 3, "rgb")
    mk = sg.limpiar_mascara(sg.mascara_hoja_por_kmeans(et, 3))
    etiq, n = ndi.label(ms)
    frag = (
        1.0
        if n == 0
        else float(
            np.bincount(etiq.ravel())[1:].max() / np.bincount(etiq.ravel())[1:].sum()
        )
    )
    return {
        "eta_gris": eg,
        "eta_sat": es,
        "iou_gs": _iou(mg, ms),
        "iou_sk": _iou(ms, mk),
        "frag": frag,
        "hoja_marco": 100 * float(ms[sg.banda_borde(g.shape)].mean()),
        "area_hoja": 100 * float(ms.mean()),
    }


def clasificar(f):
    if f["frag"] < FRAG_MIN or f["hoja_marco"] > MARCO_MAX or f["eta_sat"] < ETA_S_MIN:
        return 3
    return 1 if (f["iou_gs"] >= IOU_GS_MIN and f["eta_gris"] >= ETA_G_MIN) else 2


def hoja_contacto(rutas, archivo, titulo, cols=6, filas=4, lado=128):
    lienzo = Image.new("RGB", (cols * lado, filas * lado + 24), "white")
    ImageDraw.Draw(lienzo).text((6, 6), titulo, fill="black")
    for i, r in enumerate(rutas[: cols * filas]):
        lienzo.paste(
            Image.open(r).convert("RGB").resize((lado, lado)),
            ((i % cols) * lado, 24 + (i // cols) * lado),
        )
    lienzo.save(archivo)


def main(raiz, n_por_cat=8):
    raiz, rng, muestra = Path(raiz), random.Random(SEMILLA), []
    for cat in sorted(p for p in raiz.iterdir() if p.is_dir()):
        arch = sorted(p for p in cat.iterdir() if p.suffix.lower() in EXTS)
        muestra += rng.sample(arch, min(n_por_cat, len(arch)))
    print(f"Imágenes en la muestra: {len(muestra)}")
    filas, por_tipo = [], {1: [], 2: [], 3: []}
    for i, r in enumerate(muestra, 1):
        f = caracteristicas(sg.leer_rgb(r))
        t = clasificar(f)
        por_tipo[t].append(r)
        filas.append(
            {
                "archivo": r.name,
                "categoria": r.parent.name,
                "tipo": t,
                **{k: round(v, 3) for k, v in f.items()},
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(muestra)}")
    with open(
        "complejidad_caracteristicas.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open("complejidad_resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "imagenes", "porcentaje"])
        for t in (1, 2, 3):
            w.writerow(
                [t, len(por_tipo[t]), round(100 * len(por_tipo[t]) / len(muestra), 1)]
            )
    for t in (1, 2, 3):
        print(
            f"Tipo {t}: {len(por_tipo[t])} ({100 * len(por_tipo[t]) / len(muestra):.1f} %)"
        )
        if por_tipo[t]:
            hoja_contacto(
                rng.sample(por_tipo[t], min(24, len(por_tipo[t]))),
                f"hoja_contacto_tipo{t}.png",
                f"Tipo {t} (muestra de {len(por_tipo[t])})",
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
