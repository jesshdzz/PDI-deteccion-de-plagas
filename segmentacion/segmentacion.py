"""
Segmentación de hojas - Segundo parcial.

Algoritmos implementados desde cero con NumPy/SciPy:
  - Canny (suavizado gaussiano -> gradiente Sobel -> supresión de no máximos -> histéresis)
  - Umbralización global (algoritmo iterativo)
  - Umbralización de Otsu (un umbral)
  - K-Means (Lloyd + inicialización k-means++)
Preprocesamiento: gaussiano, mediana, gamma.
Postproceso permitido: filtro de mediana sobre la máscara binaria (sin morfología).
"""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# ----------------------------------------------------------------------
# Lectura y conversiones de color
# ----------------------------------------------------------------------
def leer_rgb(ruta):
    """Lee una imagen y la devuelve como arreglo uint8 (alto, ancho, 3) en RGB."""
    return np.asarray(Image.open(ruta).convert("RGB"))


def a_gris(rgb):
    """Escala de grises con los mismos pesos que rgb2gray de MATLAB. Devuelve uint8."""
    g = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return np.clip(np.round(g), 0, 255).astype(np.uint8)


def a_hsv(rgb):
    """RGB -> HSV con valores en [0, 1]."""
    from skimage.color import rgb2hsv

    return rgb2hsv(rgb)


def a_uint8(x):
    """Reescala un canal en [0, 1] a uint8."""
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Preprocesamiento
# ----------------------------------------------------------------------
def kernel_gaussiano(sigma):
    """Kernel 1D gaussiano normalizado, de tamaño 2*ceil(3*sigma)+1."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def gaussiano(img, sigma):
    """Filtro gaussiano separable; funciona con imágenes 2D o 3D (canal por canal)."""
    if sigma <= 0:
        return img.astype(float)
    k = kernel_gaussiano(sigma)
    out = img.astype(float)
    for eje in (0, 1):
        out = ndi.convolve1d(out, k, axis=eje, mode="nearest")
    return out


def mediana(img, n=3):
    """Filtro de mediana n x n (bordes replicados)."""
    tam = (n, n, 1) if img.ndim == 3 else (n, n)
    return ndi.median_filter(img, size=tam, mode="nearest")


def gamma(img_u8, g):
    """Corrección gamma s = 255 * (r/255)^g. g<1 aclara, g>1 oscurece."""
    return np.clip(np.round(255 * (img_u8 / 255.0) ** g), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Canny
# ----------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)


def _desplazar(a, dy, dx):
    """Devuelve b[y, x] = a[y + dy, x + dx] replicando los bordes."""
    p = np.pad(a, 1, mode="edge")
    return p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]


def canny(gris, sigma=1.5, bajo=0.10, alto=0.25, devolver_etapas=False):
    """
    Detector de bordes de Canny.

    sigma: desviación estándar del suavizado gaussiano
    bajo, alto: umbrales de la histéresis como fracción de la magnitud máxima
    """
    # Paso 1: suavizado gaussiano
    s = gaussiano(gris, sigma)
    # Paso 2: magnitud y dirección del gradiente (Sobel)
    gx = ndi.correlate(s, SOBEL_X, mode="nearest")
    gy = ndi.correlate(s, SOBEL_Y, mode="nearest")
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    # Paso 3: supresión de no máximos
    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    n1 = np.where(
        d0,
        _desplazar(mag, 0, 1),
        np.where(
            d45,
            _desplazar(mag, 1, 1),
            np.where(d90, _desplazar(mag, 1, 0), _desplazar(mag, 1, -1)),
        ),
    )
    n2 = np.where(
        d0,
        _desplazar(mag, 0, -1),
        np.where(
            d45,
            _desplazar(mag, -1, -1),
            np.where(d90, _desplazar(mag, -1, 0), _desplazar(mag, -1, 1)),
        ),
    )
    nms = np.where((mag >= n1) & (mag >= n2), mag, 0.0)
    # Paso 4: histéresis (doble umbral + conectividad 8)
    mmax = mag.max() if mag.max() > 0 else 1.0
    fuerte = nms >= alto * mmax
    debil = nms >= bajo * mmax
    etiquetas, n = ndi.label(debil, structure=np.ones((3, 3)))
    conservar = np.zeros(n + 1, bool)
    conservar[np.unique(etiquetas[fuerte])] = True
    conservar[0] = False
    bordes = conservar[etiquetas]
    if devolver_etapas:
        return bordes, {"suavizada": s, "magnitud": mag, "angulo": ang, "nms": nms}
    return bordes


# ----------------------------------------------------------------------
# Umbralización
# ----------------------------------------------------------------------
def umbral_global(img_u8, eps=0.5, T0=None):
    """Umbralización global iterativa: T = (m1 + m2) / 2 hasta que el cambio sea < eps."""
    T = float(img_u8.mean()) if T0 is None else float(T0)
    for _ in range(200):
        g1 = img_u8[img_u8 > T]
        g2 = img_u8[img_u8 <= T]
        if g1.size == 0 or g2.size == 0:
            break
        Tn = 0.5 * (g1.mean() + g2.mean())
        if abs(Tn - T) < eps:
            T = Tn
            break
        T = Tn
    return T


def otsu(img_u8):
    """Umbral de Otsu (un umbral). Devuelve (T, eta) con eta = separabilidad en [0, 1]."""
    h = np.bincount(img_u8.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mt = mu[-1]
    sb = (mt * w - mu) ** 2 / (w * (1 - w) + 1e-12)  # varianza entre clases
    T = int(np.flatnonzero(sb >= sb.max() - 1e-12).mean())
    var_total = np.sum(p * (np.arange(256) - mt) ** 2)
    eta = float(sb.max() / var_total) if var_total > 0 else 0.0
    return T, eta


# ----------------------------------------------------------------------
# K-Means (Lloyd con inicialización k-means++)
# ----------------------------------------------------------------------
def kmeans(X, k, n_init=5, max_iter=100, tol=1e-4, semilla=0):
    """
    X: (N, d) float. Devuelve (etiquetas, centros, inercia) de la mejor de n_init corridas.
    """
    rng = np.random.default_rng(semilla)
    mejor = None
    for _ in range(n_init):
        # inicialización k-means++
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(
                X[rng.choice(len(X), p=d2 / d2.sum())]
                if d2.sum() > 0
                else X[rng.integers(len(X))]
            )
        c = np.array(c, float)
        for _ in range(max_iter):
            d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
            et = d.argmin(1)
            nuevos = np.array(
                [X[et == j].mean(0) if np.any(et == j) else c[j] for j in range(k)]
            )
            if np.abs(nuevos - c).max() < tol:
                c = nuevos
                break
            c = nuevos
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        et = d.argmin(1)
        inercia = float(d[np.arange(len(X)), et].sum())
        if mejor is None or inercia < mejor[2]:
            mejor = (et, c, inercia)
    return mejor


def segmentar_kmeans(rgb, k=3, espacio="rgb", semilla=0):
    """
    Agrupa los píxeles por color. Devuelve el mapa de etiquetas (alto, ancho) y los centros.
    espacio: 'rgb' o 'hsv' (en hsv el tono se codifica como (cos, sin) para evitar el salto 0-1).
    """
    if espacio == "rgb":
        X = rgb.reshape(-1, 3).astype(float) / 255.0
    else:
        hsv = a_hsv(rgb)
        h = 2 * np.pi * hsv[..., 0]
        X = np.stack(
            [np.cos(h) * hsv[..., 1], np.sin(h) * hsv[..., 1], hsv[..., 2]], -1
        ).reshape(-1, 3)
    et, c, _ = kmeans(X, k, semilla=semilla)
    return et.reshape(rgb.shape[:2]), c


# ----------------------------------------------------------------------
# De segmentación a "máscara de hoja"
# ----------------------------------------------------------------------
def banda_borde(forma, ancho=6):
    """Máscara booleana con el marco exterior de la imagen (se asume que ahí hay fondo)."""
    m = np.zeros(forma, bool)
    m[:ancho, :] = m[-ancho:, :] = True
    m[:, :ancho] = m[:, -ancho:] = True
    return m


def mascara_hoja_por_umbral(canal_u8, T):
    """
    Binariza con T y elige automáticamente la polaridad: el fondo es la clase
    que domina el marco de la imagen; la hoja es la otra.
    """
    sup = canal_u8 > T
    borde = banda_borde(canal_u8.shape)
    return ~sup if sup[borde].mean() > 0.5 else sup


def mascara_hoja_por_kmeans(etiquetas, k):
    """Fondo = cluster dominante en el marco; hoja = unión de los demás clusters."""
    borde = banda_borde(etiquetas.shape)
    fondo = np.bincount(etiquetas[borde], minlength=k).argmax()
    return etiquetas != fondo


def limpiar_mascara(mascara, n=5):
    """Filtro de mediana sobre la máscara binaria (técnica del curso en lugar de morfología)."""
    return ndi.median_filter(mascara.astype(np.uint8), size=n, mode="nearest").astype(
        bool
    )


# ----------------------------------------------------------------------
# Visualización
# ----------------------------------------------------------------------
def superponer(rgb, mascara, color=(239, 36, 214), alfa=0.45):
    """Pinta la máscara translúcida sobre la imagen original."""
    out = rgb.astype(float).copy()
    out[mascara] = (1 - alfa) * out[mascara] + alfa * np.array(color, float)
    return out.astype(np.uint8)


def bordes_sobre(rgb, bordes, color=(239, 36, 214)):
    """Dibuja los bordes de Canny sobre la imagen."""
    out = rgb.copy()
    out[bordes] = color
    return out
