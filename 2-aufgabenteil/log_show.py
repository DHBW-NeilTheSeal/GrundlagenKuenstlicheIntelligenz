"""CSV-Auswertung der Kamera-Klassifikatoren, unabhaengig von Kamera/MediaPipe.

Im Notebook:
    from log_auswertung import auswerten
    ergebnis = auswerten('logs')

Optional konkrete Datei:
    ergebnis = auswerten(logdatei='logs/klassifikatoren_....csv')

Keine Dateien werden automatisch geschrieben. NaNs bleiben Messluecken.
Die Statistik verwendet gespeicherte Zeilen, keine zeitgewichteten Stichproben;
zwischen Klassifikationen mehrfach geloggte Kantenwerte werden mitgezaehlt.
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BEZEICHNUNGEN = {
    'auge_links_oeffnung': 'Auge links – Öffnung',
    'auge_rechts_oeffnung': 'Auge rechts – Öffnung',
    'auge_links_roh': 'Auge links – Öffnung ungeglättet',
    'auge_rechts_roh': 'Auge rechts – Öffnung ungeglättet',
    'stirn_horizontal': 'Stirn – horizontale Kanten',
    'nasenwurzel_horizontal': 'Nasenwurzel – Falten quer',
    'nasenwurzel_vertikal': 'Nasenwurzel – Falten längs',
    'wange_links_vertikal': 'Wange links – Falten längs',
    'wange_rechts_vertikal': 'Wange rechts – Falten längs',
    'kopfneigung_wert': 'Kopfneigung – normierter Wert',
    'mund_oeffnung_roh': 'Mundöffnung – ungeglättet',
    'mund_oeffnung_geglaettet': 'Mundöffnung – geglättet',
    'brille_indiz': 'Brille – Rahmenindikator (keine Wahrscheinlichkeit)',
    'brille_erkannt': 'Brille – bestätigt',
    'gesicht_erkannt': 'Gesicht – erkannt',
}
# Explizite Positivliste: nur diese Messgroessen werden angezeigt.
ANZEIGESPALTEN = tuple(BEZEICHNUNGEN)
BINAER = {'brille_erkannt', 'gesicht_erkannt'}


@dataclass
class LogAuswertung:
    datei: Path
    daten: pd.DataFrame
    statistik: pd.DataFrame
    uebersicht: pd.DataFrame
    abbildung: object


def neueste_logdatei(log_ordner='logs'):
    ordner = Path(log_ordner).expanduser().resolve()
    dateien = [p for p in ordner.glob('klassifikatoren_*.csv') if p.is_file()]
    if not dateien:
        raise FileNotFoundError(f'Keine klassifikatoren_*.csv in {ordner}')
    return max(dateien, key=lambda p: (p.stat().st_mtime_ns, p.name))


def lade_log(datei):
    datei = Path(datei).expanduser().resolve()
    try:
        daten = pd.read_csv(datei, sep=';', encoding='utf-8-sig')
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f'Logdatei enthält keine Daten: {datei.name}') from exc
    if daten.empty:
        raise ValueError(f'Logdatei enthält noch keine Messzeilen: {datei.name}')
    if 'laufzeit_s' not in daten:
        raise ValueError('Spalte laufzeit_s fehlt. Erwartet wird ein Kamera-CSV mit Semikolon-Trennung.')
    daten['laufzeit_s'] = pd.to_numeric(daten['laufzeit_s'], errors='coerce')
    daten = daten.loc[daten['laufzeit_s'].notna() & (daten['laufzeit_s'] >= 0)].copy()
    if daten.empty:
        raise ValueError('Keine gültigen Zeitwerte in laufzeit_s.')
    # Bekannte Messspalten numerisch konvertieren, auch wenn alle Werte fehlen.
    for name in BEZEICHNUNGEN:
        if name in daten:
            daten[name] = pd.to_numeric(daten[name], errors='coerce')
    # Weitere Logspalten bleiben als Rohdaten verfuegbar, werden aber nicht angezeigt.
    for name in daten:
        if name not in {'zeitstempel', 'laufzeit_s', 'frame'} and name not in BEZEICHNUNGEN:
            numeric = pd.to_numeric(daten[name], errors='coerce')
            if numeric.notna().any():
                daten[name] = numeric
    return daten.sort_values('laufzeit_s', kind='stable').reset_index(drop=True)


def messspalten(daten):
    return [name for name in ANZEIGESPALTEN if name in daten]


def beschreibe(daten, spalten):
    statistik = daten[spalten].describe().T.rename(columns={
        'count': 'Anzahl gültig', 'mean': 'Mittelwert', 'std': 'Standardabweichung',
        'min': 'Minimum', '25%': 'Q25', '50%': 'Median', '75%': 'Q75', 'max': 'Maximum',
    })
    statistik.insert(1, 'Anzahl fehlend', daten[spalten].isna().sum())
    statistik.index.name = 'Logspalte'
    statistik.insert(0, 'Messgröße', [BEZEICHNUNGEN.get(n, n) for n in statistik.index])
    return statistik


def zeichne(daten, spalten):
    fig, achsen = plt.subplots(len(spalten), 1, figsize=(12, 1.25*len(spalten)),
                              sharex=True, squeeze=False, layout='constrained')
    for ax, name in zip(achsen[:, 0], spalten):
        # Binäre Zustände als Stufen, fehlende Werte weiterhin als Lücken.
        if name in BINAER:
            ax.step(daten['laufzeit_s'], daten[name], where='post', linewidth=1)
            ax.set_yticks([0, 1])
        else:
            ax.plot(daten['laufzeit_s'], daten[name], linewidth=.9)
        ax.set_title(BEZEICHNUNGEN.get(name, name.replace('_', ' ')), loc='left', fontsize=9)
        ax.grid(alpha=.25)
        if name in BEZEICHNUNGEN:
            ax.set_ylim(-.05, 1.05)
        if daten[name].notna().sum() == 0:
            ax.text(.5, .5, 'Keine gültigen Messwerte', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='gray')
    achsen[-1, 0].set_xlabel('Zeit seit Programmstart [s]')
    return fig


def auswerten(log_ordner='logs', *, logdatei=None, anzeigen=True):
    """Neueste oder explizite Logdatei auswerten und Tabellen/Plots anzeigen.

    Rückgabe enthält Daten, Statistik, Übersicht und Figure zur Weiterverwendung.
    Fehlende Spalten alter Logs werden ausgelassen, fehlende Werte nicht ersetzt.
    """
    datei = neueste_logdatei(log_ordner) if logdatei is None else Path(logdatei).expanduser().resolve()
    daten = lade_log(datei)
    spalten = messspalten(daten)
    if not spalten:
        raise ValueError('Keine numerischen Klassifikatorwerte im Log gefunden.')
    statistik = beschreibe(daten, spalten)
    kennzahlen = {'Messzeilen': len(daten), 'Letzter Zeitwert [s]': daten['laufzeit_s'].max(),
                  'Erfasste Zeitspanne [s]': daten['laufzeit_s'].max()-daten['laufzeit_s'].min()}
    for name, label in [('gesicht_erkannt', 'Gesicht erkannt [% gültiger Zeilen]'),
                        ('brille_erkannt', 'Brille erkannt [% gültiger Zeilen]')]:
        if name in daten and daten[name].notna().any():
            kennzahlen[label] = 100*daten[name].mean()
    uebersicht = pd.DataFrame({'Wert': kennzahlen})
    fig = zeichne(daten, spalten)
    ergebnis = LogAuswertung(datei, daten, statistik, uebersicht, fig)
    if anzeigen:
        print(f'Logdatei: {datei}')
        # Ältere Logs enthalten ggf. noch Kantenwerte statt der aktuellen Faltenmerkmale.
        print('Statistik pro Logzeile; wiederholte Kantenwerte werden mitgezählt.')
        if 'brille_erkannt' not in daten:
            print('Dieses ältere Log enthält keine Brillenerkennung.')
        try:
            from IPython.display import display
        except ImportError:
            print(uebersicht.round(3).to_string())
            print(statistik.round(3).to_string())
        else:
            display(uebersicht.round(3))
            display(statistik.round(3))
        plt.show()
    return ergebnis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', default='logs', help='Ordner mit CSV-Logs')
    parser.add_argument('--datei', help='Optional konkrete CSV-Datei')
    args = parser.parse_args()
    auswerten(args.logs, logdatei=args.datei)
