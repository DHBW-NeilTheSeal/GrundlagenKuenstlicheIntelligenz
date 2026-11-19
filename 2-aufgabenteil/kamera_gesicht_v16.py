"""Kamera, geglaetteter Kopfrahmen und sieben separat auswertbare Gesichts-ROIs.

Face Landmarker liefert Augenwinkel und Gesichts-Mittellinie.
Augen und Mund werden separat zentriert; die Symmetrieachse stabilisiert das Raster. Leertaste beendet das Fenster.
"""
from pathlib import Path
import time
import sys
import os
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import urllib.request
from types import SimpleNamespace
from gesichts_klassifikatoren import (classify_regions,
                                     head_tilt, mouth_openness, MouthSmoother)

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
MODEL_PATH = Path(__file__).with_name('face_landmarker.task')
SMOOTHING = 0.18
HEAD_WIDTH_FACTOR = 1.38
HEAD_HEIGHT_FACTOR = 1.65
HEAD_SHIFT_UP = 0.13
HOLD_SECONDS = 0.65

# (Mittelpunkt x/y, Breite, Hoehe), Einheit: Augenabstand.
# Ursprung: Mitte zwischen den Augen; y positiv unterhalb der Augenlinie.
# Links/rechts bezeichnet jeweils die Bildseite (Betrachtersicht).
EYE_SMOOTHING = 0.40  # hoeher = schneller, niedriger = ruhiger
REGIONS = {
    'auge_links':        (-0.50, 0.00, 0.85, 0.52),
    'auge_rechts':       ( 0.50, 0.00, 0.85, 0.52),
    'stirn':            ( 0.00,-0.75, 1.32, 0.525),
    'nasenwurzel':      ( 0.00, 0.18, 0.28, 0.42),
    'nasolabial_links': (-0.38, 0.68, 0.34, 0.46),
    'nasolabial_rechts':( 0.38, 0.68, 0.34, 0.46),
    'mund':             ( 0.00, 1.02, 1.00, 0.55),
}
ROI_COLOR = (0, 220, 255)


# Augenoeffnung aus der Lidspalte, unabhaengig von Hough-Kreisstaerke.
# Links/rechts beziehen sich weiterhin auf die Bildseite.
EYE_LAYOUTS = (
    {'corners': (33,133), 'upper': (160,159,158), 'lower': (144,145,153),
     'contour': (33,160,159,158,133,153,145,144)},
    {'corners': (362,263), 'upper': (385,386,387), 'lower': (380,374,373),
     'contour': (362,385,386,387,263,373,374,380)},
)


@dataclass
class EyeOpeningResult:
    opening: float
    raw_opening: float
    ratio: float
    calibrated: bool
    contour: np.ndarray
    pairs: np.ndarray


def eye_geometry(points, layout):
    """Drei Lidabstaende senkrecht zur Augenlinie / Augenbreite.

    Projektion statt euklidischem Abstand: seitlich versetzte Landmarken
    erhoehen die Oeffnung nicht kuenstlich. Roll- und skalierungsinvariant,
    aber starke Perspektive/Verdeckung bleibt eine Einschraenkung.
    """
    corners = points[list(layout['corners'])]
    vector = corners[1]-corners[0]
    width = float(np.linalg.norm(vector))
    contour = points[list(layout['contour'])]
    upper, lower = points[list(layout['upper'])],points[list(layout['lower'])]
    if not np.isfinite(np.vstack([corners,upper,lower])).all() or width<12:
        return None
    normal = np.array([-vector[1],vector[0]])/width
    gaps = abs((lower-upper)@normal)/width
    # Mittlere Lidspalte erhaelt etwas mehr Gewicht.
    ratio = float(np.dot(gaps,[0.25,0.50,0.25]))
    if not 0<=ratio<=0.65:
        return None
    return {'ratio': ratio, 'contour': contour.copy(),
            'pairs': np.stack([upper,lower],axis=1)}


class EyeOpeningTracker:
    """Schneller Lidklassifikator mit expliziter, getrennter Kalibrierung.

    Standard: Oeffnungsgrenze 0.38 statt 0.28, um bei normal geoeffneten
    Augen Reserve fuer weites Aufreissen zu lassen.
    Standardwerte sind Startwerte, keine allgemeingueltigen anatomischen
    Grenzen. O: 1.5s weit geoeffnet halten. C: 1.5s geschlossen halten.
    Median je Auge; beide Grenzwerte muessen mindestens 0.06 auseinanderliegen.
    Keine automatische Anpassung, die laengeren Lidschluss als offen lernt.
    """
    def __init__(self):
        self.names=('auge_links','auge_rechts')
        self.reset()

    def reset(self):
        self.bounds={n:{'closed':0.035,'open':0.38} for n in self.names}
        self.calibrated={n:set() for n in self.names}
        self.values={}; self.times={}
        self.pending=None
        self.message='O: weit offen kalibrieren | C: geschlossen | R: Reset'

    def begin(self, mode, now):
        self.pending={'mode':mode,'start':now,'samples':{n:[] for n in self.names}}

    def status(self, now):
        if self.pending:
            mode='WEIT OFFEN' if self.pending['mode']=='open' else 'GESCHLOSSEN'
            remaining=max(0,1.5-(now-self.pending['start']))
            return f'Augen {mode} halten: {remaining:.1f}s'
        return self.message

    def update(self, features, now):
        if self.pending:
            elapsed=now-self.pending['start']
            if 0.30<=elapsed<=1.5:
                for name,feature in features.items():
                    if feature is not None:
                        self.pending['samples'][name].append(feature['ratio'])
            if elapsed>=1.5:
                mode=self.pending['mode']; accepted=[]
                for name,samples in self.pending['samples'].items():
                    if len(samples)<5:
                        continue
                    candidate=float(np.median(samples))
                    other=self.bounds[name]['closed' if mode=='open' else 'open']
                    if (candidate-other if mode=='open' else other-candidate)<0.06:
                        continue
                    self.bounds[name][mode]=candidate
                    self.calibrated[name].add(mode)
                    self.values.pop(name,None); self.times.pop(name,None)
                    accepted.append(name)
                self.message=('Kalibrierung '+mode+': '+', '.join(accepted) if accepted else
                              'Kalibrierung ungueltig. Frontal ausrichten und wiederholen.')
                print(self.message)
                self.pending=None
        result={}
        for name in self.names:
            feature=features.get(name)
            if feature is None:
                self.values.pop(name,None); self.times.pop(name,None)
                continue
            low,high=self.bounds[name]['closed'],self.bounds[name]['open']
            raw=float(np.clip((feature['ratio']-low)/(high-low),0,1))
            previous=self.values.get(name)
            dt=now-self.times.get(name,now)
            if previous is None or dt>HOLD_SECONDS:
                value=raw
            else:
                # Schliessen schnell, Oeffnen etwas ruhiger; keine 5-Hz-Begrenzung.
                tau=0.025 if raw<previous else 0.075
                alpha=1-np.exp(-max(0,dt)/tau)
                value=previous+alpha*(raw-previous)
            if raw==0:
                value=0.0
            self.values[name]=value; self.times[name]=now
            result[name]=EyeOpeningResult(float(value),raw,feature['ratio'],
                                         len(self.calibrated[name])==2,
                                         feature['contour'],feature['pairs'])
        return result


# Spezialisierte Falten- und Brillenmerkmale; keine weiteren Modelle notwendig.
DETAIL_REGIONS = {'nasenwurzel','nasolabial_links','nasolabial_rechts'}


@dataclass
class RidgeResult:
    value: float
    fraction: float
    pixel_count: int
    total_pixels: int
    mask: np.ndarray


def line_kernel(length, angle_degrees):
    length=max(3,int(length)|1)
    kernel=np.zeros((length,length),np.uint8)
    c=(length-1)/2
    angle=np.deg2rad(angle_degrees)
    delta=np.array([np.cos(angle),np.sin(angle)])*c
    cv2.line(kernel,tuple(np.rint(c-delta).astype(int)),
             tuple(np.rint(c+delta).astype(int)),1,1)
    return kernel


def wrinkle_ridges(image, orientation='vertical', exclusion=None):
    """Mehrskalige dunkle Rinnen mit beidseitig hellerem Umfeld.

    Keine lokale Histogrammspreizung und kein pro Frame angepasster Kontrast-
    Schwellwert: Hautrauschen wird dadurch nicht automatisch zu einer Falte.
    Hessian-Eigenwerte pruefen Linienform/Richtung, flankierende Abtastungen
    verwerfen einfache Helligkeitsstufen. Mindestens zwei Skalen muessen passen.
    Die Werte beschreiben sichtbare Linienstruktur, keine reale Faltentiefe.
    """
    h,w=image.shape[:2]
    empty=np.zeros((h,w),np.uint8)
    if min(h,w)<18:
        return RidgeResult(0.,0.,0,0,empty)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    scale=min(1.,160/max(h,w))
    if scale<1:
        gray=cv2.resize(gray,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    gray=gray.astype(np.float32)
    gh,gw=gray.shape
    yy,xx=np.mgrid[:gh,:gw].astype(np.float32)
    valid=np.ones((gh,gw),bool)
    # Randkonturen (Nase/Lippen/Brauen) nicht als Falte bewerten.
    my,mx=max(3,round(gh*.12)),max(3,round(gw*.12))
    valid[:my]=False;valid[-my:]=False;valid[:,:mx]=False;valid[:,-mx:]=False
    if exclusion is not None:
        valid &= cv2.resize(exclusion,(gw,gh),interpolation=cv2.INTER_NEAREST)==0
    strength=np.zeros_like(gray)
    votes=np.zeros_like(gray,np.uint8)
    tangent_target=np.pi/2 if orientation=='vertical' else 0.
    normal_target=tangent_target+np.pi/2
    for sigma in (.9,1.5,2.3):
        blurred=cv2.GaussianBlur(gray,(0,0),sigma)
        dxx=cv2.Sobel(blurred,cv2.CV_32F,2,0,ksize=3)*(sigma*sigma/4.)
        dyy=cv2.Sobel(blurred,cv2.CV_32F,0,2,ksize=3)*(sigma*sigma/4.)
        dxy=cv2.Sobel(blurred,cv2.CV_32F,1,1,ksize=3)*(sigma*sigma/4.)
        delta=np.sqrt((dxx-dyy)**2+4*dxy*dxy)
        major=(dxx+dyy+delta)/2.
        minor=(dxx+dyy-delta)/2.
        coherence=1-np.clip(abs(minor)/np.maximum(abs(major),1e-5),0,1)
        angle=.5*np.arctan2(2*dxy,dxx-dyy)
        alignment=abs(np.cos(angle-normal_target))
        nx,ny=np.cos(angle),np.sin(angle)
        offset=2.5*sigma
        flank_a=cv2.remap(blurred,xx+nx*offset,yy+ny*offset,cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        flank_b=cv2.remap(blurred,xx-nx*offset,yy-ny*offset,cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        contrast_a=flank_a-blurred;contrast_b=flank_b-blurred
        contrast=np.minimum(contrast_a,contrast_b)
        # Absolute Schwellwerte in 8-Bit-Graustufen, nicht an Rauschen normiert.
        candidate=(major>.7)&(coherence>.55)&(contrast>=3.0)&(
            alignment>=np.cos(np.deg2rad(35)))&valid
        votes+=candidate.astype(np.uint8)
        strength=np.maximum(strength,np.where(candidate,contrast*coherence,0))
    binary=((votes>=2)&valid).astype(np.uint8)*255
    # Kurze Unterbrechungen verbinden, keine starre Gerade erzwingen:
    # Nasolabialfalten sind typischerweise gebogen.
    connected=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    connected[~valid]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(connected,8)
    selected=np.zeros_like(binary)
    for index in range(1,count):
        ys,xs=np.nonzero(labels==index)
        if len(xs)<6:continue
        coords=np.column_stack([xs,ys]).astype(float)
        covariance=np.cov(coords,rowvar=False)
        eig=np.linalg.eigvalsh(covariance)
        elongation=np.sqrt(max(eig[-1],0)/max(eig[0],.2))
        span=np.ptp(ys) if orientation=='vertical' else np.ptp(xs)
        required=max(7,round((gh if orientation=='vertical' else gw)*.18))
        if span>=required and elongation>=2.4:
            selected[labels==index]=255
    selected[~valid]=0
    mask=cv2.resize(selected,(w,h),interpolation=cv2.INTER_NEAREST)
    original_valid=cv2.resize(valid.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    pixels=int(np.count_nonzero(mask));total=int(np.count_nonzero(original_valid))
    fraction=pixels/max(1,total)
    ridge_strength=float(np.median(strength[(selected>0)&(votes>=2)])) if pixels else 0.
    # Sowohl Linienausdehnung als auch Kontrast zaehlen. Kein Rauschen->1-Autoscaling.
    value=float(np.clip(np.sqrt(fraction/.10)*ridge_strength/12.,0,1))
    return RidgeResult(value,fraction,pixels,total,mask)


class ScalarSmoother:
    def __init__(self,tau=.25):
        self.tau=tau;self.previous={};self.times={}

    def update(self,key,value,now):
        last=self.times.get(key,now)
        old=self.previous.get(key)
        if old is None or now-last>HOLD_SECONDS:
            filtered=float(value)
        else:
            filtered=old+(1-np.exp(-max(0,now-last)/self.tau))*(value-old)
        self.previous[key]=filtered;self.times[key]=now
        return float(filtered)


@dataclass
class GlassesResult:
    score: float
    detected: bool
    label: str
    frame_mask: np.ndarray


def glasses_frame_evidence(frame,refs,eye_features):
    """Klassische Heuristik: zwei Rahmen plus verbindender Nasensteg.

    Kein trainierter Brillenklassifikator und keine kalibrierte Wahrscheinlichkeit.
    Transparente/randlose Fassungen koennen fehlen; markante Schatten oder
    Augenbrauen koennen Fehler erzeugen. Augenkonturen werden ausgeschlossen.
    """
    left,right=refs[:2]
    distance=float(np.linalg.norm(right-left))
    if distance<35 or any(eye_features.get(n) is None for n in ('auge_links','auge_rechts')):
        return None
    horizontal=(right-left)/distance
    vertical=np.array([-horizontal[1],horizontal[0]])
    center=(left+right)/2
    corners=(center+np.array([[-1.05,-.35],[1.05,-.35],[-1.05,.40]])
             @np.stack([horizontal,vertical])*distance).astype(np.float32)
    h,w=96,240
    transform=cv2.getAffineTransform(corners,np.float32([[0,0],[w-1,0],[0,h-1]]))
    # Keine schwarzen Bildraender als Rahmenkante bewerten.
    image_h,image_w=frame.shape[:2]
    fourth=corners[1]+corners[2]-corners[0]
    all_corners=np.vstack([corners,fourth])
    if (all_corners[:,0].min()<0 or all_corners[:,1].min()<0 or
        all_corners[:,0].max()>=image_w or all_corners[:,1].max()>=image_h):
        return None
    crop=cv2.warpAffine(frame,transform,(w,h))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    edges=cv2.Canny(gray,35,90)
    exclude=np.zeros_like(edges)
    for feature in eye_features.values():
        if feature is not None:
            polygon=feature['contour']@transform[:,:2].T+transform[:,2]
            cv2.fillPoly(exclude,[np.rint(polygon).astype(np.int32)],255)
    exclude=cv2.dilate(exclude,np.ones((5,5),np.uint8))
    edges[exclude>0]=0
    support=cv2.dilate(edges,np.ones((3,3),np.uint8))>0
    # Standardisierte Positionen anhand derselben affinen Abbildung.
    eyes=np.vstack([left,right])@transform[:,:2].T+transform[:,2]
    d=float(np.linalg.norm(eyes[1]-eyes[0]))
    rim_scores=[];rim_masks=[]
    for eye in eyes:
        best=0.;best_mask=np.zeros_like(edges)
        for half_width in (.30,.36,.42):
            for half_height in (.18,.23,.29):
                theta=np.linspace(0,2*np.pi,96,endpoint=False)
                xs=np.rint(eye[0]+d*half_width*np.cos(theta)).astype(int)
                ys=np.rint(eye[1]+d*half_height*np.sin(theta)).astype(int)
                inside=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
                hits=np.zeros(96,dtype=bool)
                hits[inside]=support[ys[inside],xs[inside]]
                quadrants=[float(part.mean()) for part in np.array_split(hits,4)]
                # Augenbraue oder einzelner Lidbogen allein reicht nicht.
                coverage=float(hits.mean())
                supported=sum(q>=.40 for q in quadrants)
                score=np.clip((coverage-.25)/.55,0,1) if supported>=3 and min(quadrants)>=.25 else 0.
                if score>best:
                    best=float(score)
                    best_mask[:]=0;best_mask[ys[inside],xs[inside]]=255
        rim_scores.append(best);rim_masks.append(best_mask)
    # Bruecke: kurze horizontale Struktur zwischen inneren Linsenraendern.
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    bridge=np.zeros_like(edges)
    mid=eyes.mean(axis=0)
    bx0,bx1=round(mid[0]-.18*d),round(mid[0]+.18*d)
    by0,by1=round(mid[1]-.13*d),round(mid[1]+.10*d)
    bridge[by0:by1,bx0:bx1]=255
    horizontal=(edges>0)&(abs(dy)>1.5*abs(dx))&(bridge>0)
    band=horizontal[by0:by1,bx0:bx1]
    bridge_score=float(np.any(band,axis=0).mean()) if band.size else 0.
    bilateral=min(rim_scores)
    score=float(np.clip(.75*bilateral+.25*bridge_score,0,1))
    # Leichtere Fassungen zulassen; minimale beidseitige Form und Bruecke
    # bleiben Voraussetzung. Die eigentliche Entscheidung wird zeitlich bestaetigt.
    if bilateral<.30 or bridge_score<.15:
        score=min(score,.30)
    # Maske nur plausibler, beidseitiger Rahmen; spaeter aus Nasen-ROI ausschliessen.
    candidate=(edges>0)&(cv2.dilate(np.maximum(rim_masks[0],rim_masks[1]),
                                  np.ones((5,5),np.uint8))>0)
    candidate|=horizontal
    inv=cv2.invertAffineTransform(transform)
    frame_mask=cv2.warpAffine(candidate.astype(np.uint8)*255,inv,
                             (image_w,image_h),flags=cv2.INTER_NEAREST)
    return score,frame_mask


class GlassesTracker:
    """Konservative Entscheidung mit zeitlicher Bestaetigung und Hysterese."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.score=0.;self.detected=False;self.last=None
        self.high_since=None;self.low_since=None;self.high_samples=0
        self.history=[]

    def update(self,evidence,now):
        if evidence is None:
            if self.last is None or now-self.last>HOLD_SECONDS:self.reset()
            return None
        raw,mask=evidence
        if self.last is None or now-self.last>HOLD_SECONDS:
            self.reset()
            dt=.20
        else:
            dt=max(0.,now-self.last)
        self.last=now
        self.history.append(float(raw));self.history=self.history[-5:]
        median=float(np.median(self.history))
        self.score+=(1-np.exp(-dt/.50))*(median-self.score)
        # Einzelne Treffer oder nur zwei gute Frames loesen nichts aus.
        if raw>=.48 and self.score>=.44:
            if self.high_since is None:self.high_since=now
            self.high_samples+=1
            if now-self.high_since>=.80 and self.high_samples>=5:
                self.detected=True
        else:
            self.high_since=None;self.high_samples=0
        if raw<.20 or self.score<.30:
            if self.low_since is None:self.low_since=now
            if now-self.low_since>=.60:self.detected=False
        else:
            self.low_since=None
        label=('Brille erkannt (stabil)' if self.detected else
               'Brille nicht bestaetigt' if self.score>.30 else 'Keine Brille erkannt')
        return GlassesResult(float(self.score),self.detected,label,mask)


def landmark_face(landmarks, frame_shape):
    """Praezise Augenwinkel statt grober BlazeFace-Augen-Keypoints."""
    h, w = frame_shape[:2]
    points = np.array([[p.x*w, p.y*h] for p in landmarks])
    # Augenmittelpunkte aus innerem und aeusserem Augenwinkel.
    eyes = np.array([points[[33, 133]].mean(axis=0),
                     points[[362, 263]].mean(axis=0)])
    order = np.argsort(eyes[:,0])
    eyes = eyes[order]
    eye_features = dict(zip(('auge_links','auge_rechts'),
                           [eye_geometry(points,EYE_LAYOUTS[i]) for i in order]))
    refs = np.vstack([eyes, points[168], points[[13, 14]].mean(axis=0),
                      points[10], points[152]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=lo[0], origin_y=lo[1],
                                     width=hi[0]-lo[0], height=hi[1]-lo[1]),
        refs=refs, eye_features=eye_features,
        details=points[[168,9,98,61,327,291]].copy(),
        mouth_contour=points[[78,191,80,81,82,13,312,311,310,415,
                              308,324,318,402,317,14,87,178,88,95]])


def measured_eyes(face, frame_shape):
    """Referenzen: Augen, Nasenwurzel, Mund, Stirnmitte, Kinn."""
    refs = face.refs
    if not np.isfinite(refs).all() or np.linalg.norm(refs[1]-refs[0]) < 8:
        return None
    return refs.copy()


def get_region_polygons(refs,details=None):
    """Symmetrieachse stabilisiert die Neigung; Augen/Mund separat zentriert.

    Stirn-Kinn ist eine angenaeherte 2D-Symmetrieachse, keine 3D-Pose.
    """
    left, right, nose, mouth, forehead, chin = refs
    distance = np.linalg.norm(right-left)
    origin = (left+right)/2
    eye_vertical = np.array([-(right-left)[1], (right-left)[0]]) / distance
    candidates = [(chin-forehead, 0.55), (mouth-origin, 0.25),
                  (mouth-nose, 0.05), (eye_vertical, 0.15)]
    vertical = np.zeros(2)
    for vector, weight in candidates:
        length = np.linalg.norm(vector)
        if length > 1e-6:
            unit = vector/length
            if unit[1] < 0:
                unit = -unit
            vertical += weight*unit
    angle = np.arctan2(-vertical[0], vertical[1])
    # Kleine Schaetzfehler nicht als dauerhafte Schraeglage anzeigen.
    if abs(angle) < np.deg2rad(4):
        angle = 0.0
    vertical = np.array([-np.sin(angle), np.cos(angle)]) * distance
    horizontal = np.array([vertical[1], -vertical[0]])
    centers = {'auge_links': left, 'auge_rechts': right,
               'mund': mouth, 'nasenwurzel': nose - 0.12*vertical}
    polygons = {}
    for name, (x, y, w, h) in REGIONS.items():
        center = centers.get(name, origin+x*horizontal+y*vertical)
        corners = np.array([[-w/2,-h/2], [w/2,-h/2],
                            [w/2,h/2], [-w/2,h/2]])
        polygons[name] = (center + corners[:, :1]*horizontal
                         + corners[:, 1:]*vertical).astype(np.float32)
    if details is not None:
        def rectangle(center,horizontal_unit,vertical_unit,width,height):
            offsets=np.array([[-width/2,-height/2],[width/2,-height/2],
                              [width/2,height/2],[-width/2,height/2]])
            return (center+offsets[:,0,None]*horizontal_unit
                    +offsets[:,1,None]*vertical_unit).astype(np.float32)
        hu,vu=horizontal/distance,vertical/distance
        root,brow,nose_l,mouth_l,nose_r,mouth_r=details
        # Glabella zwischen den Brauen, nicht die untere Nase.
        root_center=.60*root+.40*brow-.04*vertical
        polygons['nasenwurzel']=rectangle(root_center,hu,vu,.48*distance,.58*distance)
        sides=sorted([(nose_l,mouth_l),(nose_r,mouth_r)],key=lambda pair:pair[0][0])
        for index,(nose_side,mouth_side) in enumerate(sides):
            sign=-1 if index==0 else 1
            # Auf die Haut seitlich der Nasen-/Lippenkontur verschieben.
            upper=nose_side+sign*.14*distance*hu
            lower=mouth_side+sign*.14*distance*hu
            direction=lower-upper
            length=float(np.linalg.norm(direction))
            local_v=direction/max(length,1e-6) if length>1 else vu
            if np.dot(local_v,vu)<0:local_v=-local_v
            local_h=np.array([local_v[1],-local_v[0]])
            center=.48*upper+.52*lower
            height=np.clip(length*1.50,.60*distance,1.00*distance)
            name='nasolabial_links' if index==0 else 'nasolabial_rechts'
            polygons[name]=rectangle(center,local_h,local_v,.48*distance,height)
    return polygons


def extract_regions(frame, polygons):
    """Unbeschriftete, aufrechte BGR-Ausschnitte; Rand-ROIs auslassen.

    region_images[name] bleibt ein separat auswertbares NumPy-Array.
    """
    h, w = frame.shape[:2]
    images = {}
    for name, corners in polygons.items():
        if (corners[:, 0].min() < 0 or corners[:, 1].min() < 0
                or corners[:, 0].max() > w-1 or corners[:, 1].max() > h-1):
            continue
        rw = max(2, int(round(np.linalg.norm(corners[1]-corners[0]))))
        rh = max(2, int(round(np.linalg.norm(corners[3]-corners[0]))))
        target = np.float32([[0, 0], [rw-1, 0], [0, rh-1]])
        transform = cv2.getAffineTransform(corners[[0, 1, 3]], target)
        images[name] = cv2.warpAffine(frame, transform, (rw, rh))
    return images


PANEL_WIDTH = 330
ROW_HEIGHT = 112


def sobel_edges(image):
    """Sobel-Gradientenbetrag, fuer die Anzeige pro ROI auf 0..255 skaliert.

    Filter auf dem Originalausschnitt anwenden, erst danach verkleinern.
    Fuer quantitative Auswertung ggf. den unskalierten Betrag verwenden.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    peak = float(magnitude.max())
    if peak <= 1e-6:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip(magnitude * (255.0/peak), 0, 255).astype(np.uint8)


def draw_edge_panel(display, region_images, left, measurements, tilt_result=None,glasses_result=None):
    """Alle sieben ROIs in fester Reihenfolge untereinander anzeigen."""
    display[:, left:] = (28, 28, 28)
    for index, name in enumerate(REGIONS):
        top = index*ROW_HEIGHT
        suffix=' (Falten)' if name in DETAIL_REGIONS else ' (Sobel)'
        cv2.putText(display, name + suffix, (left+10, top+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, ROI_COLOR, 1)
        image = region_images.get(name)
        if image is None:
            cv2.putText(display, 'Nicht verfuegbar', (left+10, top+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
            continue
        edges = sobel_edges(image)
        values = measurements.get(name, {})
        if name in DETAIL_REGIONS:
            edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
            for kind,color in [('horizontal',(255,190,80)),('vertical',(210,120,255))]:
                ridge=values.get(kind)
                if ridge is not None and ridge.mask.shape==edges.shape[:2]:
                    edges[ridge.mask>0]=color
        eye = values.get('eye')
        if eye is not None:
            edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR) if edges.ndim==2 else edges
            contour = values.get('eye_contour_roi')
            if contour is not None:
                cv2.polylines(edges,[np.rint(contour).astype(np.int32)],True,(0,255,0),1)
        max_w, max_h = 140, ROW_HEIGHT-28
        ih, iw = edges.shape[:2]
        scale = min(max_w/iw, max_h/ih)
        tw, th = max(1, round(iw*scale)), max(1, round(ih*scale))
        thumb = cv2.resize(edges, (tw, th),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x = left+10+(max_w-tw)//2
        y = top+23+(max_h-th)//2
        display[y:y+th, x:x+tw] = (cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR) if thumb.ndim==2 else thumb)
        values = measurements.get(name, {})
        horizontal = values.get('horizontal')
        vertical = values.get('vertical')
        bars = []
        if horizontal is not None:
            bars.append(('Falten quer' if name in DETAIL_REGIONS else 'Horiz. Kanten',horizontal.value,(255,190,80)))
        if vertical is not None:
            bars.append(('Falten laengs' if name in DETAIL_REGIONS else 'Vert. Kanten',vertical.value,(210,120,255)))
        eye = values.get('eye')
        if eye is not None:
            bars.append(('Auge offen',eye.opening,(100,230,100)))
            cv2.putText(display,f'Lid/Breite: {eye.ratio:.3f}',(left+162,top+82),
                        cv2.FONT_HERSHEY_SIMPLEX,0.34,(210,210,210),1)
            cv2.putText(display,'kalibriert' if eye.calibrated else 'Standardwerte',
                        (left+162,top+103),cv2.FONT_HERSHEY_SIMPLEX,0.34,(210,210,210),1)
        mouth = values.get('mouth')
        if mouth is not None:
            bars.append(('Mund offen',mouth,(100,200,255)))
        for bar_index, (label,value,color) in enumerate(bars):
            draw_value_bar(display,left+162,top+39+bar_index*37,label,value,color)

    top = len(REGIONS)*ROW_HEIGHT
    cv2.putText(display, 'Kopfneigung', (left+10,top+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,ROI_COLOR,1)
    angle_text = ('Nicht verfuegbar' if tilt_result is None else
                  f'Augenlinie: {tilt_result.angle_degrees:+.1f} Grad')
    cv2.putText(display,angle_text,(left+10,top+44),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(230,230,230),1)
    draw_value_bar(display,left+162,top+76,'Neigung',
                   None if tilt_result is None else tilt_result.value,(180,220,100))

    top=(len(REGIONS)+1)*ROW_HEIGHT
    cv2.putText(display,'Brillenerkennung',(left+10,top+18),
                cv2.FONT_HERSHEY_SIMPLEX,.40,ROI_COLOR,1)
    label='Nicht verfuegbar' if glasses_result is None else glasses_result.label
    color=(100,240,100) if glasses_result and glasses_result.detected else (210,210,210)
    cv2.putText(display,label,(left+10,top+45),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    draw_value_bar(display,left+162,top+76,'Rahmen-Indiz',
                   None if glasses_result is None else glasses_result.score,(100,220,230))


def mouth_contour_in_roi(contour, corners, image_shape):
    """Gleiche affine Abbildung wie beim Ausschneiden des Mundfensters."""
    h,w = image_shape[:2]
    target=np.float32([[0,0],[w-1,0],[0,h-1]])
    transform=cv2.getAffineTransform(corners[[0,1,3]],target)
    return (np.asarray(contour)@transform[:,:2].T+transform[:,2]).astype(np.float32)


def draw_value_bar(display, x, y, label, value, color):
    text = label + (' --' if value is None else f' {value:.2f}')
    cv2.putText(display, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (230,230,230), 1)
    width, height = PANEL_WIDTH-174, 11
    cv2.rectangle(display, (x,y+5), (x+width,y+5+height), (65,65,65), -1)
    if value is not None:
        filled = int(round(width*np.clip(value,0,1)))
        if filled:
            cv2.rectangle(display, (x,y+5), (x+filled,y+5+height), color, -1)


def analyze_regions(region_images):
    """Die unveraenderten Kantenklassifikatoren fuer Nicht-Augen-ROIs."""
    # Die Augen werden separat in jedem Frame mit Lidlandmarks bewertet.
    non_eye={n:image for n,image in region_images.items() if not n.startswith('auge_')}
    return classify_regions(non_eye)


class ClassifierCSVLogger:
    """Eine CSV-Zeile pro Kameraframe, nur Messwerte, keine Bilder.

    Nasenwurzel/Wangen verwenden ab v15 mehrskalige Rinnenfilter statt allgemeiner
    Kantendichte; alte Spaltennamen bleiben fuer Notebook-Kompatibilitaet.
    brille_indiz ist ein heuristischer Merkmalswert, keine Wahrscheinlichkeit.

    Kanten werden ca. alle 0.2s neu berechnet; Augen in jedem Frame.
    kanten_kreis_alter_s bleibt aus Kompatibilitaet erhalten und bezeichnet
    jetzt nur das Alter der Kantenwerte.
    Fehlende Werte sind leer, echte Nullmessungen werden als 0 geschrieben.
    UTF-8 mit BOM, Semikolon-Trennung, Dezimalpunkt.
    """
    FIELDS = [
        'zeitstempel', 'laufzeit_s', 'frame', 'gesicht_erkannt',
        'kanten_kreis_alter_s',
        'auge_links_oeffnung', 'auge_rechts_oeffnung',
        'auge_links_roh', 'auge_rechts_roh',
        'auge_links_lidverhaeltnis', 'auge_rechts_lidverhaeltnis',
        'auge_links_kalibriert', 'auge_rechts_kalibriert',
        'stirn_horizontal', 'nasenwurzel_horizontal',
        'nasenwurzel_vertikal', 'wange_links_vertikal', 'wange_rechts_vertikal',
        'kopfneigung_wert', 'kopfneigung_grad',
        'mund_oeffnung_roh', 'mund_oeffnung_geglaettet',
        'brille_indiz','brille_erkannt',
    ]

    def __init__(self, directory=None):
        directory = (Path(__file__).resolve().parent/'logs' if directory is None
                     else Path(directory))
        directory.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        self.path = directory/f'klassifikatoren_{stamp}.csv'
        self.file = self.path.open('x',newline='',encoding='utf-8-sig')
        self.writer = csv.DictWriter(self.file,fieldnames=self.FIELDS,delimiter=';')
        try:
            self.writer.writeheader()
            self.file.flush()
        except BaseException:
            self.file.close()
            raise
        self.start = time.monotonic()
        self.last_flush = self.start
        self.frame = 0
        self.rows = 0

    @staticmethod
    def number(value):
        return '' if value is None else f'{float(value):.6f}'

    def log(self, now, measurements, tilt_result, face_detected,
            last_classified=None, mouth_raw=None,glasses_result=None):
        self.frame += 1
        row = dict.fromkeys(self.FIELDS,'')
        row.update(zeitstempel=datetime.now().astimezone().isoformat(timespec='milliseconds'),
                   laufzeit_s=self.number(max(0,now-self.start)),
                   frame=self.frame,gesicht_erkannt=int(face_detected))
        mapping = [
            ('auge_links_oeffnung','auge_links','eye','opening'),
            ('auge_rechts_oeffnung','auge_rechts','eye','opening'),
            ('stirn_horizontal','stirn','horizontal','value'),
            ('nasenwurzel_horizontal','nasenwurzel','horizontal','value'),
            ('nasenwurzel_vertikal','nasenwurzel','vertical','value'),
            ('wange_links_vertikal','nasolabial_links','vertical','value'),
            ('wange_rechts_vertikal','nasolabial_rechts','vertical','value'),
        ]
        if face_detected:
            if last_classified is not None and last_classified>0:
                row['kanten_kreis_alter_s']=self.number(max(0,now-last_classified))
            for field,region,kind,attribute in mapping:
                result=measurements.get(region,{}).get(kind)
                if result is not None:
                    row[field]=self.number(getattr(result,attribute))
            for name in ('auge_links','auge_rechts'):
                eye=measurements.get(name,{}).get('eye')
                if eye is not None:
                    row[name+'_roh']=self.number(eye.raw_opening)
                    row[name+'_lidverhaeltnis']=self.number(eye.ratio)
                    row[name+'_kalibriert']=int(eye.calibrated)
            if tilt_result is not None:
                row['kopfneigung_wert']=self.number(tilt_result.value)
                row['kopfneigung_grad']=self.number(tilt_result.angle_degrees)
            if glasses_result is not None:
                row['brille_indiz']=self.number(glasses_result.score)
                row['brille_erkannt']=int(glasses_result.detected)
            row['mund_oeffnung_roh']=self.number(mouth_raw)
            row['mund_oeffnung_geglaettet']=self.number(
                measurements.get('mund',{}).get('mouth'))
        self.writer.writerow(row)
        self.rows += 1
        # Zwischensichern, um bei einem spaeteren Fehler wenig zu verlieren.
        if now-self.last_flush>=1.0:
            self.file.flush()
            self.last_flush=now

    def close(self):
        if not self.file.closed:
            self.file.close()  # flush aller noch gepufferten Zeilen
            print(f'CSV-Log gespeichert: {self.path} ({self.rows} Zeilen)')


def resolve_camera_platform(platform_mode=None):
    """Prioritaet: Parameter > CAMERA_PLATFORM > automatische Erkennung."""
    mode = platform_mode if platform_mode is not None else os.environ.get('CAMERA_PLATFORM', 'auto')
    mode = str(mode).strip().lower() or 'auto'
    aliases = {'win': 'windows', 'win32': 'windows',
               'macos': 'mac', 'darwin': 'mac'}
    mode = aliases.get(mode, mode)
    if mode not in ('auto', 'windows', 'mac'):
        raise ValueError(
            f'Ungueltige Kameraplattform {mode!r}. Erlaubt: auto, windows, mac.'
        )
    if mode == 'auto':
        if sys.platform == 'win32':
            return 'windows'
        if sys.platform == 'darwin':
            return 'mac'
    return mode


def open_camera(index=0, platform_mode=None):
    """Windows: DirectShow; Mac: AVFoundation; sonst OpenCV-Auswahl."""
    if not isinstance(index, int) or index < 0:
        raise ValueError('Der Kameraindex muss eine nichtnegative ganze Zahl sein.')
    mode = resolve_camera_platform(platform_mode)
    if mode == 'windows':
        backend = cv2.CAP_DSHOW
        backend_name = 'DirectShow'
    elif mode == 'mac':
        backend = cv2.CAP_AVFOUNDATION
        backend_name = 'AVFoundation'
    else:
        backend = cv2.CAP_ANY
        backend_name = 'automatische OpenCV-Auswahl'
    print(f'Oeffne Kamera {index}, Plattform={mode}, Backend={backend_name} ...', flush=True)
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f'Kamera {index} konnte mit {backend_name} nicht geoeffnet werden. '
            'Plattformauswahl, Kameraindex und Berechtigung pruefen.'
        )
    print('Kamera geoeffnet.', flush=True)
    return camera


def main(platform_mode=None, camera_index=0):
    platform_mode = resolve_camera_platform(platform_mode)
    if not MODEL_PATH.exists():
        print('Lade Gesichtsmodell einmalig herunter ...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera = open_camera(camera_index, platform_mode)

    smoothed = None
    eyes_smoothed = None
    last_eyes_seen = 0.0
    last_seen = 0.0
    last_timestamp = -1
    last_classified = 0.0
    cached_measurements = {}
    mouth_filter = MouthSmoother(tau=0.25)
    eye_tracker = EyeOpeningTracker()
    details_smoothed=None
    glasses_tracker=GlassesTracker()
    detail_smoother=ScalarSmoother(.40)
    glasses_result=None

    logger = None
    try:
        logger = ClassifierCSVLogger()
        print(f'CSV-Logging gestartet: {logger.path}')
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None or frame.size == 0:
                    raise RuntimeError(
                        'Kamera liefert keine Bilddaten. Andere Kamera-Apps schliessen '
                        'und den Kamerazugriff pruefen.'
                    )
                height, width = frame.shape[:2]
                clean_frame = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                result = SimpleNamespace(detections=[
                    landmark_face(lm, frame.shape) for lm in result.face_landmarks])
                now = time.monotonic()

                eyes = None
                if now-last_eyes_seen > HOLD_SECONDS:
                    cached_measurements = {}
                    last_classified = 0.0
                    details_smoothed=None
                    glasses_result=None
                if result.detections:
                    face = max(result.detections,
                               key=lambda d: d.bounding_box.width * d.bounding_box.height)
                    box = face.bounding_box
                    measured = np.array([box.origin_x, box.origin_y,
                                         box.width, box.height], dtype=np.float64)
                    if smoothed is None or now - last_seen > HOLD_SECONDS:
                        smoothed = measured
                    else:
                        smoothed = (1.0 - SMOOTHING) * smoothed + SMOOTHING * measured
                    if details_smoothed is None or now-last_seen>HOLD_SECONDS:
                        details_smoothed=face.details.copy()
                    else:
                        alpha_details=1-np.exp(-max(.001,now-last_seen)/.10)
                        details_smoothed+=alpha_details*(face.details-details_smoothed)
                    last_seen = now
                    eyes = measured_eyes(face, frame.shape)
                    if eyes is not None:
                        if eyes_smoothed is None or now-last_eyes_seen > HOLD_SECONDS:
                            eyes_smoothed = eyes
                        else:
                            # Zeitabhaengig: gleiche Glaettung bei verschiedenen FPS.
                            dt = max(0.001, now-last_eyes_seen)
                            alpha = 1.0 - (1.0-EYE_SMOOTHING) ** (dt*30.0)
                            eyes_smoothed += alpha * (eyes-eyes_smoothed)
                        last_eyes_seen = now

                region_images = {}
                measurements = {}
                tilt_result = None
                mouth_raw = None
                eye_results=eye_tracker.update(face.eye_features if eyes is not None else {},now)
                info = 'Kein Gesicht erkannt'
                if smoothed is not None and now - last_seen <= HOLD_SECONDS:
                    x, y, w, h = smoothed
                    cx, cy = int(round(x + w / 2)), int(round(y + h / 2))
                    head_w, head_h = w * HEAD_WIDTH_FACTOR, h * HEAD_HEIGHT_FACTOR
                    head_cx = x + w / 2
                    head_cy = y + h / 2 - HEAD_SHIFT_UP * h
                    x1 = max(0, int(round(head_cx - head_w / 2)))
                    y1 = max(0, int(round(head_cy - head_h / 2)))
                    x2 = min(width - 1, int(round(head_cx + head_w / 2)))
                    y2 = min(height - 1, int(round(head_cy + head_h / 2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    if eyes_smoothed is not None and now-last_eyes_seen <= HOLD_SECONDS:
                        polygons = get_region_polygons(eyes_smoothed,details_smoothed)
                        if eyes is not None:
                            tilt_result = head_tilt(eyes_smoothed[:2])
                        # Bei Aussetzern nur Anzeige halten, keine alten ROIs auswerten.
                        if eyes is not None and result.detections:
                            region_images = extract_regions(clean_frame, polygons)
                            # Maximal fuenf Klassifikationen/Sekunde; Livebild laeuft weiter.
                            if now-last_classified >= 0.20:
                                cached_measurements = analyze_regions(
                                    {n:im for n,im in region_images.items() if n not in DETAIL_REGIONS})
                                # Brillenrahmen koennen Faltenkanten ueberdecken.
                                evidence=glasses_frame_evidence(clean_frame,eyes_smoothed,face.eye_features)
                                glasses_result=glasses_tracker.update(evidence,now)
                                for detail_name in DETAIL_REGIONS:
                                    if detail_name not in region_images:continue
                                    image=region_images[detail_name]
                                    exclusion=None
                                    if glasses_result is not None and glasses_result.detected:
                                        h_roi,w_roi=image.shape[:2]
                                        transform=cv2.getAffineTransform(polygons[detail_name][[0,1,3]],
                                            np.float32([[0,0],[w_roi-1,0],[0,h_roi-1]]))
                                        exclusion=cv2.warpAffine(glasses_result.frame_mask,transform,
                                                               (w_roi,h_roi),flags=cv2.INTER_NEAREST)
                                        exclusion=cv2.dilate(exclusion,np.ones((3,3),np.uint8))
                                    kinds=('horizontal','vertical') if detail_name=='nasenwurzel' else ('vertical',)
                                    values={}
                                    for kind in kinds:
                                        ridge=wrinkle_ridges(image,kind,exclusion)
                                        ridge.value=detail_smoother.update((detail_name,kind),ridge.value,now)
                                        values[kind]=ridge
                                    cached_measurements[detail_name]=values
                                last_classified = now
                            measurements = {name: cached_measurements[name]
                                            for name in region_images
                                            if name in cached_measurements}
                        for name,eye_result in eye_results.items():
                            if name in region_images:
                                measurements[name]={'eye':eye_result,
                                    'eye_contour_roi': mouth_contour_in_roi(
                                        eye_result.contour,polygons[name],region_images[name].shape)}
                                cv2.polylines(frame,[np.rint(eye_result.contour).astype(np.int32)],
                                              True,(0,255,0),1)
                                for upper,lower in eye_result.pairs:
                                    cv2.line(frame,tuple(np.rint(upper).astype(int)),
                                             tuple(np.rint(lower).astype(int)),(255,220,0),1)
                        if eyes is not None and 'mund' in region_images:
                            contour = mouth_contour_in_roi(face.mouth_contour,
                                                           polygons['mund'],region_images['mund'].shape)
                            mouth_result = mouth_openness(region_images['mund'],contour)
                            mouth_raw = mouth_result.value
                            # Nur echte aktuelle Messungen filtern, keine gehaltenen Frames.
                            filtered = mouth_filter.update(mouth_result.value,now)
                            measurements.setdefault('mund',{})['mouth'] = filtered
                        for name, corners in polygons.items():
                            points = np.rint(corners).astype(np.int32)
                            cv2.polylines(frame, [points], True, ROI_COLOR, 1)
                            rx, ry = points[0]
                            cv2.putText(frame, name, (max(0, rx), max(12, ry-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROI_COLOR, 1)
                        axis_start = tuple(np.rint(eyes_smoothed[4]).astype(int))
                        axis_end = tuple(np.rint(eyes_smoothed[5]).astype(int))
                        cv2.line(frame, axis_start, axis_end, (255, 180, 0), 1)
                        for eye in eyes_smoothed[:2]:
                            cv2.circle(frame, tuple(np.rint(eye).astype(int)),
                                       3, (0, 255, 0), -1)
                    else:
                        eyes_smoothed = None
                    info = f'Gesichtsmittelpunkt: X = {cx} px    Y = {cy} px'
                elif now - last_seen > HOLD_SECONDS:
                    smoothed = None
                    eyes_smoothed = None

                logger.log(now,measurements,tilt_result,eyes is not None,
                           last_classified=last_classified,mouth_raw=mouth_raw,
                           glasses_result=glasses_result if eyes is not None else None)
                display_height = max(height+100, (len(REGIONS)+2)*ROW_HEIGHT)
                display = np.zeros((display_height, width+PANEL_WIDTH, 3), dtype=np.uint8)
                display[:height, :width] = frame
                cv2.putText(display, info, (12, height + 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, 'Space: Ende | O: offen | C: zu | R: Reset', (12, height + 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
                cv2.putText(display,eye_tracker.status(now),(12,height+80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.40,(180,230,180),1)
                draw_edge_panel(display, region_images, width, measurements, tilt_result,
                                glasses_result if eyes is not None else None)
                cv2.imshow('Kamera - Gesichtsregionen', display)
                key=cv2.waitKey(1) & 0xFF
                if key in (32,27):
                    break
                if key in (ord('o'),ord('O')):
                    eye_tracker.begin('open',time.monotonic())
                elif key in (ord('c'),ord('C')):
                    eye_tracker.begin('closed',time.monotonic())
                elif key in (ord('r'),ord('R')):
                    eye_tracker.reset()
    finally:
        try:
            if logger is not None:
                logger.close()
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('auto', 'windows', 'mac'), default=None,
                        help='Kameraplattform; sonst CAMERA_PLATFORM oder auto')
    parser.add_argument('--camera', type=int, default=0, help='Kameraindex (Standard: 0)')
    args = parser.parse_args()
    main(platform_mode=args.platform, camera_index=args.camera)
