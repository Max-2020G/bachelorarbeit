import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------
# Ein-/Ausschalter fuer die beiden Batterie-Messreihen.
# Auf False setzen, um die jeweilige Messreihe KOMPLETT zu ueberspringen:
# kein Laden der Daten, kein Fit, kein Einzelplot, und sie taucht auch im
# gemeinsamen Plot am Ende nicht mehr auf.
# -----------------------------------------------------------------------
PLOT_20S = True
PLOT_60S = True

# Alle PDFs dieses Skripts landen gesammelt in diesem Unterordner.
OUTDIR = "pdf/time_constant"
os.makedirs(OUTDIR, exist_ok=True)


def x_from_filename(file):
    """
    Liest aus einem Dateinamen wie "100ms.txt" oder "1,5s_20s_Messung.txt"
    die Zeitkonstante heraus und gibt sie in Sekunden (float) zurueck.

    Ablauf:
    1. Nur den reinen Dateinamen behalten (ohne Ordnerpfad davor) und die
       Endung ".txt" abschneiden.
    2. Nur den ERSTEN Teil vor einem "_" behalten. Bei den normalen
       with/without-Illumination-Dateien gibt es kein "_" im Namen, das
       aendert also nichts. Bei den Batterie-Dateien steht danach aber noch
       ein Suffix wie "_20s_Messung", das uns hier nicht interessiert
       (z.B. wird aus "1,5s_20s_Messung" nur "1,5s").
    3. Deutsches Dezimalkomma "," durch einen Punkt "." ersetzen -- Python
       versteht bei float() nur den Punkt als Dezimaltrennzeichen, die
       Batterie-Dateinamen sind aber mit Komma geschrieben (z.B. "1,5s").
    4. Mit einem regulaeren Ausdruck Zahl und Einheit (ms oder s) trennen.
       "([\\d.]+)" faengt die Ziffern/Punkte (die Zahl), "(ms|s)?" optional
       die Einheit dahinter.
    5. Ist die Einheit "ms", durch 1000 teilen, um Sekunden zu bekommen.
    """
    name = file.split("/")[-1].removesuffix(".txt")
    token = name.split("_")[0].replace(",", ".")
    value, unit = re.match(r"([\d.]+)(ms|s)?$", token).groups()
    return float(value) / 1000 if unit == "ms" else float(value)


def load(folder):
    """
    Laedt alle .txt-Messdateien aus "folder" und berechnet pro Datei:
      - x:   die Zeitkonstante (aus dem Dateinamen, siehe x_from_filename)
      - U:   den Mittelwert der gemessenen Spannung ueber die ganze Messung
      - std: die Standardabweichung der gemessenen Spannung

    Gibt drei NumPy-Arrays (x, U, std) zurueck, aufsteigend nach x sortiert
    (glob.glob findet Dateien nur in Dateisystem-/alphabetischer, nicht in
    numerischer Reihenfolge, daher das Sortieren am Ende).
    """
    x, U, std = [], [], []
    for file in sorted(glob.glob(f"{folder}/*.txt")):
        # Jede Datei hat oben ein paar Kommentarzeilen (beginnend mit "%")
        # und danach zwei Spalten "Zeit; Amplitude", durch ";" getrennt.
        # Die Zeit-Spalte der Rohmessung brauchen wir hier nicht (wir wollen
        # ja nur den Mittelwert/die Streuung der Amplitude), daher "_".
        _, u = np.genfromtxt(file, delimiter=";", comments="%", unpack=True)
        x.append(x_from_filename(file))
        U.append(u.mean())
        std.append(u.std())

    order = np.argsort(x)
    return np.array(x)[order], np.array(U)[order], np.array(std)[order]


def model(x, A, b):
    """Fitmodell: Rauschamplitude A/sqrt(x) plus konstanter Untergrund b."""
    return A / np.sqrt(x) + b


def plot_single(x_plot, U, std, x_fit_plot, y_fit, xlabel, filename):
    """
    Zeichnet EINEN Einzelplot (Messpunkte mit Fehlerbalken + Fitkurve) und
    speichert ihn ab. Wird unten zweimal pro Messreihe aufgerufen: einmal mit
    x = 1/sqrt(tau) (ergibt eine Gerade), einmal mit x = tau selbst (ergibt
    die abfallende Kurve, an der man "mehr tau -> weniger Rauschen" direkt
    ablesen kann).

    Wichtig: diese Funktion transformiert selbst nichts. "x_plot"/"x_fit_plot"
    sind bereits die fertigen Werte fuer die x-Achse (z.B. schon 1/sqrt(tau)
    gerechnet), "y_fit" sind die fertig mit model() berechneten y-Werte der
    Fitkurve. So kann dieselbe Funktion fuer beide Varianten (linear und
    gegen tau) benutzt werden, ohne dass sie wissen muss, welche Transformation
    gerade verwendet wird.
    """
    plt.errorbar(x_plot, U * 10**6, yerr=2 * std * 10**6, fmt="o",
                 capsize=3, color="#639A00", ecolor="#82C80097")
    plt.plot(x_fit_plot, y_fit * 10**6, color="#FF9100")
    plt.xlabel(xlabel)
    plt.ylabel(r"U [$\mu V$]")
    plt.grid()
    plt.savefig(filename)
    plt.close()


# -----------------------------------------------------------------------
# Daten laden. Vier Messreihen insgesamt:
#   wi  = with Illumination     (Belichtung an)
#   oi  = without Illumination  (Belichtung aus)
#   b20 = Batterie-Messung, Lock-In-Zeitkonstante ueber 20s gemessen
#   b60 = Batterie-Messung, Lock-In-Zeitkonstante ueber 60s gemessen
# Die Batterie-Messreihen werden nur geladen, wenn ihr Schalter oben auf
# True steht.
# -----------------------------------------------------------------------
x_wi, U_wi, std_wi = load("data/with_Illumination/with_Illumination")
x_oi, U_oi, std_oi = load("data/without_Illuminitation/without_Illuminitation")

if PLOT_20S:
    x_b20, U_b20, std_b20 = load("data/Batterie/20s_Messungen")
if PLOT_60S:
    x_b60, U_b60, std_b60 = load("data/Batterie/60s_Messungen")

# print(U_wi)
# print(U_oi)

# -----------------------------------------------------------------------
# Fits: jede Messreihe bekommt ihren eigenen Fit von model(x, A, b).
# curve_fit(model, x, U) gibt ein Tupel (popt, pcov) zurueck:
#   popt = die gefundenen besten Parameter, hier [A, b]
#   pcov = die Kovarianzmatrix des Fits -- die Wurzel der Diagonalelemente
#          ist die Unsicherheit des jeweiligen Parameters:
#          sqrt(pcov[0, 0]) -> Unsicherheit von A, sqrt(pcov[1, 1]) -> von b
# -----------------------------------------------------------------------
popt_wi, pcov_wi = curve_fit(model, x_wi, U_wi)
popt_oi, pcov_oi = curve_fit(model, x_oi, U_oi)
print(f"A_wi = {popt_wi[0]:.3e} +- {np.sqrt(pcov_wi[0, 0]):.3e}")
print(f"A_oi = {popt_oi[0]:.3e} +- {np.sqrt(pcov_oi[0, 0]):.3e}")
print(f"b_wi = {popt_wi[1]:.3e} +- {np.sqrt(pcov_wi[1, 1]):.3e}")
print(f"b_oi = {popt_oi[1]:.3e} +- {np.sqrt(pcov_oi[1, 1]):.3e}")

if PLOT_20S:
    popt_b20, pcov_b20 = curve_fit(model, x_b20, U_b20)
    print(f"A_b20 = {popt_b20[0]:.3e} +- {np.sqrt(pcov_b20[0, 0]):.3e}")
    print(f"b_b20 = {popt_b20[1]:.3e} +- {np.sqrt(pcov_b20[1, 1]):.3e}")

if PLOT_60S:
    popt_b60, pcov_b60 = curve_fit(model, x_b60, U_b60)
    print(f"A_b60 = {popt_b60[0]:.3e} +- {np.sqrt(pcov_b60[0, 0]):.3e}")
    print(f"b_b60 = {popt_b60[1]:.3e} +- {np.sqrt(pcov_b60[1, 1]):.3e}")

# -----------------------------------------------------------------------
# Fuer die Fit-Kurven im Plot brauchen wir viele, dicht liegende x-Werte
# zwischen dem kleinsten und groessten gemessenen x -- sonst wuerde die
# Kurve nur an den 13 Messpunkten ausgewertet und eckig aussehen.
# -----------------------------------------------------------------------
x_fit_wi = np.linspace(x_wi.min(), x_wi.max(), 1000)
x_fit_oi = np.linspace(x_oi.min(), x_oi.max(), 1000)
if PLOT_20S:
    x_fit_b20 = np.linspace(x_b20.min(), x_b20.max(), 1000)
if PLOT_60S:
    x_fit_b60 = np.linspace(x_b60.min(), x_b60.max(), 1000)


# -----------------------------------------------------------------------
# Einzelplots: fuer jede Messreihe ZWEI Varianten:
#
# a) "linear" gegen 1/sqrt(tau): Modell U = A/sqrt(tau) + b wird mit
#    x = 1/sqrt(tau) zu U = A*x + b, also einer echten Geraden. Zeigt, ob
#    die Messpunkte wirklich der 1/sqrt(tau)-Rauschgesetzmaessigkeit folgen,
#    und der Wert bei x=0 (tau -> unendlich) ist direkt der Rauschuntergrund
#    b. Achtung: dadurch ist die x-Achse "umgedreht" -- kleine tau liegen
#    rechts, tau -> unendlich liegt links bei x=0.
#
# b) gegen tau selbst (keine Transformation): dieselbe Kurve, aber in der
#    "natuerlichen" Richtung -- man liest direkt ab, dass das Rauschen mit
#    steigender Zeitkonstante abnimmt (fallende Kurve von links nach rechts).
#
# y-Achse ist in beiden Faellen in Mikrovolt (*10**6), da die Rohwerte in
# Volt im Bereich weniger 1e-7 V liegen und so schlecht lesbar waeren.
# -----------------------------------------------------------------------
plot_single(1 / np.sqrt(x_wi), U_wi, std_wi,
            1 / np.sqrt(x_fit_wi), model(x_fit_wi, *popt_wi),
            r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_wi.pdf")
plot_single(x_wi, U_wi, std_wi,
            x_fit_wi, model(x_fit_wi, *popt_wi),
            r"$\tau$ [s]", f"{OUTDIR}/time_constant_wi_vs_tau.pdf")

plot_single(1 / np.sqrt(x_oi), U_oi, std_oi,
            1 / np.sqrt(x_fit_oi), model(x_fit_oi, *popt_oi),
            r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_oi.pdf")
plot_single(x_oi, U_oi, std_oi,
            x_fit_oi, model(x_fit_oi, *popt_oi),
            r"$\tau$ [s]", f"{OUTDIR}/time_constant_oi_vs_tau.pdf")

if PLOT_20S:
    plot_single(1 / np.sqrt(x_b20), U_b20, std_b20,
                1 / np.sqrt(x_fit_b20), model(x_fit_b20, *popt_b20),
                r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_b20.pdf")
    plot_single(x_b20, U_b20, std_b20,
                x_fit_b20, model(x_fit_b20, *popt_b20),
                r"$\tau$ [s]", f"{OUTDIR}/time_constant_b20_vs_tau.pdf")

if PLOT_60S:
    plot_single(1 / np.sqrt(x_b60), U_b60, std_b60,
                1 / np.sqrt(x_fit_b60), model(x_fit_b60, *popt_b60),
                r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_b60.pdf")
    plot_single(x_b60, U_b60, std_b60,
                x_fit_b60, model(x_fit_b60, *popt_b60),
                r"$\tau$ [s]", f"{OUTDIR}/time_constant_b60_vs_tau.pdf")

# -----------------------------------------------------------------------
# Gemeinsamer Plot: alle aktuell eingeschalteten Messreihen zusammen, mit
# je einer eigenen Farbe. Auch hier beide Varianten (1/sqrt(tau) und tau).
# Es werden bewusst nur die Messpunkte (keine Fitkurven) gezeigt, um den
# Plot nicht zu ueberladen.
# -----------------------------------------------------------------------
def plot_combined(x_wi_plot, x_oi_plot, x_b20_plot, x_b60_plot, xlabel, filename):
    """Zeichnet den gemeinsamen Plot fuer eine gegebene x-Achsen-Variante."""
    plt.errorbar(x_wi_plot, U_wi * 10**6, yerr=2 * std_wi * 10**6, fmt="o",
                 capsize=3, label="with Illumination", ecolor="#82C80097", color="#639A00")
    plt.errorbar(x_oi_plot, U_oi * 10**6, yerr=2 * std_oi * 10**6, fmt="o",
                 capsize=3, label="without Illumination", color="#FF9100", ecolor="#FF91009D")
    if PLOT_20S:
        plt.errorbar(x_b20_plot, U_b20 * 10**6, yerr=2 * std_b20 * 10**6, fmt="o",
                     capsize=3, label="Batterie, 20s Messung", color="#3366CC", ecolor="#3366CC97")
    if PLOT_60S:
        plt.errorbar(x_b60_plot, U_b60 * 10**6, yerr=2 * std_b60 * 10**6, fmt="o",
                     capsize=3, label="Batterie, 60s Messung", color="#9933CC", ecolor="#9933CC97")
    plt.xlabel(xlabel)
    plt.ylabel(r"U [$\mu V$]")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


plot_combined(
    1 / np.sqrt(x_wi), 1 / np.sqrt(x_oi),
    1 / np.sqrt(x_b20) if PLOT_20S else None,
    1 / np.sqrt(x_b60) if PLOT_60S else None,
    r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_combined.pdf",
)
plot_combined(
    x_wi, x_oi,
    x_b20 if PLOT_20S else None,
    x_b60 if PLOT_60S else None,
    r"$\tau$ [s]", f"{OUTDIR}/time_constant_combined_vs_tau.pdf",
)
