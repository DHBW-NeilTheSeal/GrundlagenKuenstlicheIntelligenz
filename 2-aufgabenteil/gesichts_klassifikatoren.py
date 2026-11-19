"""Einfache Bildmerkmale fuer Gesichts-ROIs (BGR oder Grau, uint8).

Kreiswert: normalisierter Kreis-Hough-Akkumulator, keine Wahrscheinlichkeit
und kein physiologisch kalibrierter Augenoeffnungsgrad. Auch andere runde
Objekte koennen einen hohen Wert erzeugen. Iris/Pupille werden nicht semantisch
unterschieden. Horizontale Kanten: Anteil duenn ausgeduennter Kantenpixel,
bei 25 Prozent auf 1 gesaettigt. Schwellen sind fuer 8-Bit-Bilder gedacht.
"""
from dataclasses import dataclass, replace
import cv2
import numpy as np


@dataclass(frozen=True)
class CircleResult:
    opening: float
    accumulator: int
    perimeter_pixels: int
    circle: tuple | None  # (x, y, radius) im Eingabebild
    raw_support: float = 0.0
    lid_factor: float = 1.0


@dataclass(frozen=True)
class HorizontalResult:
    value: float
    fraction: float
    pixel_count: int
    total_pixels: int
    mask: np.ndarray


def grayscale(image):
    if image is None or image.size == 0:
        raise ValueError('Leerer Bildausschnitt')
    if image.dtype != np.uint8:
        raise ValueError('uint8-Bild mit Intensitaeten von 0 bis 255 erwartet')
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError('Graubild oder BGR-Bild erwartet')


def circle_openness(image, min_radius=None, max_radius=None,
                    canny_low=35, canny_high=90, min_support=0.50,
                    max_side=64):
    """Diskrete Kreis-Hough-Transformation mit explizitem Akkumulator.

    Fuer jeden Radius stimmen Kantenpixel fuer moegliche Kreismittelpunkte.
    A_r(x,y) zaehlt unterstuetzte Positionen der diskreten Kreislinie.
    Ein Pixel Toleranz gleicht Rasterung und Canny-Lokalisierung aus.
    Wert = max A_r / Anzahl Kreislinienpixel; unter min_support -> 0.
    Die Gradientenrichtung muss zur radialen Richtung passen (45 Grad).
    Ganze Kreise und Zentren im mittleren ROI-Bereich werden bevorzugt:
    Mittelpunkte sind auf die inneren 60 Prozent je Achse beschraenkt.
    min/max_radius in Pixeln des Eingabebildes, intern ggf. verkleinert.
    """
    if not 0 <= min_support <= 1 or max_side < 16:
        raise ValueError('Ungueltige Kreisparameter')
    gray = grayscale(image)
    original_h, original_w = gray.shape
    scale = min(1.0, max_side/max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(original_w*scale)),
                                max(1, round(original_h*scale))),
                          interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    if min(h, w) < 9:
        return CircleResult(0.0, 0, 0, None)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.dilate(cv2.Canny(blur, canny_low, canny_high),
                       np.ones((3,3), np.uint8)) > 0
    dx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    nx = dx/np.maximum(magnitude, 1e-6)
    ny = dy/np.maximum(magnitude, 1e-6)
    rmin = max(3, round(min_radius*scale) if min_radius is not None
               else round(h*0.10))
    rmax = min((min(h,w)-1)//2,
               round(max_radius*scale) if max_radius is not None else round(h*0.45))
    best_score, best_votes, best_count, best_circle = 0.0, 0, 0, None
    for radius in range(rmin, rmax+1):
        angles = np.linspace(0, 2*np.pi, max(32, round(8*np.pi*radius)), endpoint=False)
        offsets = np.unique(np.rint(np.column_stack(
            (radius*np.cos(angles), radius*np.sin(angles)))).astype(int), axis=0)
        # Jede diskrete Kreisposition stimmt hoechstens einmal pro Zentrum.
        accumulator = np.zeros((h,w), dtype=np.uint16)
        for ox, oy in offsets:
            x0, x1 = max(0,-ox), min(w,w-ox)
            y0, y1 = max(0,-oy), min(h,h-oy)
            radial = abs(nx[y0+oy:y1+oy,x0+ox:x1+ox]*ox
                         + ny[y0+oy:y1+oy,x0+ox:x1+ox]*oy)
            aligned = radial >= 0.7071*np.hypot(ox,oy)
            accumulator[y0:y1,x0:x1] += (
                edges[y0+oy:y1+oy,x0+ox:x1+ox] & aligned)
        # Keine angeschnittenen Kreise, keine Randstrukturen als Kandidaten.
        x0, x1 = max(radius, round(w*.20)), min(w-radius, round(w*.80))
        y0, y1 = max(radius, round(h*.20)), min(h-radius, round(h*.80))
        if x0 >= x1 or y0 >= y1:
            continue
        sub = accumulator[y0:y1,x0:x1]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        votes = int(sub[iy,ix]); count = len(offsets)
        score = votes/count
        cx, cy = x0+ix, y0+iy
        ox, oy = offsets[:,0], offsets[:,1]
        supported = edges[cy+oy,cx+ox] & (
            abs(nx[cy+oy,cx+ox]*ox + ny[cy+oy,cx+ox]*oy)
            >= 0.7071*np.hypot(ox,oy))
        # Ein Lidbogen/eine Gerade darf nicht als voller Kreis gelten.
        sectors = ((np.arctan2(oy,ox)+2*np.pi) % (2*np.pi)
                   / (np.pi/2)).astype(int)
        coverage = [supported[sectors==q].mean() for q in range(4)]
        if min(coverage) < 0.20:
            continue
        if score > best_score:
            best_score, best_votes, best_count = score, votes, count
            best_circle = ((x0+ix)/scale, (y0+iy)/scale, radius/scale)
    if best_score < min_support:
        return CircleResult(0.0, best_votes, best_count, None)
    return CircleResult(float(np.clip(best_score,0,1)), best_votes,
                        best_count, best_circle, best_score)


def horizontal_edges(image, canny_low=35, canny_high=90,
                     angle_tolerance_degrees=20.0, saturation_fraction=0.25):
    """Zaehlt Canny-Kantenpixel mit vorwiegend vertikalem Sobel-Gradienten.

    Horizontale KANTE bedeutet vertikaler GRADIENT (dy, nicht dx).
    value = min(1, (pixel_count/total_pixels)/saturation_fraction).
    Die Maske ist ausgeduennt, sodass breite Sobel-Baender nicht doppelt zaehlen.
    """
    if not 0 < angle_tolerance_degrees < 90 or not 0 < saturation_fraction <= 1:
        raise ValueError('Ungueltige Kantenparameter')
    gray = grayscale(image)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    dx = cv2.Sobel(blur, cv2.CV_32F, 1,0,ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_32F, 0,1,ksize=3)
    thin_edges = cv2.Canny(blur, canny_low, canny_high) > 0
    mask = thin_edges & (abs(dy) > 0) & (
        abs(dx) <= abs(dy)*np.tan(np.deg2rad(angle_tolerance_degrees)))
    count = int(np.count_nonzero(mask)); total = gray.size
    fraction = count/total
    return HorizontalResult(min(1.0, fraction/saturation_fraction),
                            fraction, count, total, mask.astype(np.uint8)*255)


def vertical_edges(image, canny_low=35, canny_high=90,
                   angle_tolerance_degrees=20.0, saturation_fraction=0.25):
    """Vertikale Kanten haben einen horizontalen Sobel-Gradienten.

    Gleiche Normierung wie horizontal_edges; maske in Originalorientierung.
    """
    gray = grayscale(image)
    transposed = horizontal_edges(gray.T.copy(), canny_low, canny_high,
                                  angle_tolerance_degrees, saturation_fraction)
    return HorizontalResult(transposed.value, transposed.fraction,
                            transposed.pixel_count, transposed.total_pixels,
                            transposed.mask.T.copy())


CIRCLE_REGIONS = {'auge_links', 'auge_rechts'}
HORIZONTAL_REGIONS = {'stirn', 'nasenwurzel'}
VERTICAL_REGIONS = {'nasenwurzel', 'nasolabial_links', 'nasolabial_rechts'}


def classify_regions(region_images, eye_openings=None):
    """Selektive Klassifikatoren. eye_openings: Lidabstand / Augenbreite.

    Lidfaktor 0 bei <=0.10, 1 bei >=0.24, linear dazwischen.
    Diese Schwellen sind Startwerte und im Versuch zu kalibrieren.
    Kreiswert = Hough-Unterstuetzung * Lidfaktor. Fehlt die Lidmessung,
    bleibt der reine Kreiswert erhalten (fuer Einzelbildexperimente).
    """
    result = {}
    for name, image in region_images.items():
        circle = None
        if name in CIRCLE_REGIONS:
            ratio = None if eye_openings is None else eye_openings.get(name)
            factor = 1.0 if ratio is None else float(np.clip((ratio-0.10)/0.14,0,1))
            if factor == 0:
                circle = CircleResult(0.0,0,0,None,0.0,0.0)
            else:
                circle = circle_openness(image)
                circle = replace(circle, opening=circle.opening*factor,
                                 lid_factor=factor)
        result[name] = {
            'horizontal': horizontal_edges(image) if name in HORIZONTAL_REGIONS else None,
            'vertical': vertical_edges(image) if name in VERTICAL_REGIONS else None,
            'circle': circle,
        }
    return result


@dataclass(frozen=True)
class HeadTiltResult:
    value: float
    angle_degrees: float  # Vorzeichen in Bildkoordinaten


def head_tilt(eyes, dead_zone_degrees=float(np.degrees(np.arctan(0.10))),
              full_tilt_degrees=30.0):
    """0 bei waagerechter Augenlinie; sanfter Anstieg bis 1 bei 30 Grad.

    10% Steigung = atan(0.1) = 5.71 Grad, bis dahin bleibt der Wert 0.
    Falls 10 Grad gemeint sind: dead_zone_degrees=10 setzen.
    Smoothstep vermeidet Spruenge an den beiden Grenzwerten.
    """
    if not 0 <= dead_zone_degrees < full_tilt_degrees <= 90:
        raise ValueError('Ungueltige Neigungsgrenzen')
    eyes = np.asarray(eyes,dtype=float)
    if eyes.shape != (2,2) or not np.isfinite(eyes).all():
        raise ValueError('Zwei gueltige Augenmittelpunkte erwartet')
    eyes = eyes[np.argsort(eyes[:,0])]
    dx,dy = eyes[1]-eyes[0]
    if np.hypot(dx,dy) < 1e-6:
        raise ValueError('Augenmittelpunkte fallen zusammen')
    angle = float(np.degrees(np.arctan2(dy,dx)))
    t = float(np.clip((abs(angle)-dead_zone_degrees)
                      /(full_tilt_degrees-dead_zone_degrees),0,1))
    return HeadTiltResult(t*t*(3-2*t),angle)


@dataclass(frozen=True)
class MouthResult:
    value: float
    fraction: float
    pixel_count: int
    total_pixels: int
    mask: np.ndarray


def mouth_openness(image, inner_lip_contour, full_fraction=0.90):
    """Anteil der Mundoeffnung am Mund-ROI, aus inneren Lippenlandmarks.

    Kontur in Pixelkoordinaten DIESES Ausschnitts. Damit werden dunkle
    Lippen/Bartschatten nicht als offene Mundhoehle klassifiziert und Zaehne
    zaehlen ebenfalls zur Oeffnung. 90% ROI-Flaeche entsprechen Wert 1.
    Geometrischer Schaetzer, nicht physiologisch kalibriert.
    """
    if not 0 < full_fraction <= 1:
        raise ValueError('Ungueltiger Flaechenanteil')
    gray = grayscale(image)
    contour = np.asarray(inner_lip_contour,dtype=np.float32)
    if contour.ndim!=2 or contour.shape[1]!=2 or len(contour)<3 or not np.isfinite(contour).all():
        raise ValueError('Gueltige innere Lippenkontur erwartet')
    area = float(cv2.contourArea(contour))
    mask = np.zeros(gray.shape,np.uint8)
    # Ein geschlossener Mund als Linie soll nicht wegen Rasterung Pixel sammeln.
    if area >= 1.0:
        cv2.fillPoly(mask,[np.rint(contour).astype(np.int32)],255)
    count = int(np.count_nonzero(mask)); fraction = count/gray.size
    return MouthResult(min(1.0,fraction/full_fraction),fraction,count,gray.size,mask)


class MouthSmoother:
    """Zeitabhaengiger Tiefpass: tau=0.25s, unabhaengig von Bildrate.

    Nach einer Erkennungspause >0.65s wird mit dem neuen Wert initialisiert.
    """
    def __init__(self,tau=0.25,reset_after=0.65):
        if tau<=0 or reset_after<=0:
            raise ValueError('Zeitkonstanten muessen positiv sein')
        self.tau=tau; self.reset_after=reset_after
        self.value=None; self.last_time=None

    def update(self,value,timestamp):
        value=float(np.clip(value,0,1))
        if self.last_time is not None and timestamp<self.last_time:
            raise ValueError('Zeitstempel muessen monoton sein')
        if self.value is None or timestamp-self.last_time>self.reset_after:
            self.value=value
        else:
            alpha=1-np.exp(-(timestamp-self.last_time)/self.tau)
            self.value += alpha*(value-self.value)
        self.last_time=timestamp
        return float(self.value)
