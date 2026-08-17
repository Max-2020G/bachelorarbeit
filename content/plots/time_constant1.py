import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# plt.rcParams["figure.figsize"] = (2.62, 2.0)  # Breite, Höhe in Zoll
# plt.rcParams["font.size"] = 11
# -----------------------------------------------------------------------
# Ein-/Ausschalter fuer die beiden Batterie-Messreihen.
# Auf False setzen, um die jeweilige Messreihe KOMPLETT zu ueberspringen:
# kein Laden der Daten, kein Fit, kein Einzelplot, und sie taucht auch im
# gemeinsamen Plot am Ende nicht mehr auf.
# -----------------------------------------------------------------------
PLOT_20S = True
PLOT_60S = True

# -----------------------------------------------------------------------
# Maske pro Messreihe: Indizes der Messpunkte, die NICHT in den jeweiligen
# Fit eingehen sollen. Die Punkte sind nach Zeitkonstante aufsteigend
# sortiert (siehe load()), Index 0 ist also der Punkt mit der kleinsten
# Zeitkonstante. Leere Liste [] = alle Punkte werden verwendet.
# Ausgeschlossene Punkte werden automatisch grau im Plot dargestellt,
# fliessen aber nicht in curve_fit() ein.
# -----------------------------------------------------------------------
EXCLUDE_WI = []
EXCLUDE_OI = []
EXCLUDE_B20 = [0]  # kleinste Zeitkonstante hat einen sehr grossen Fehler
EXCLUDE_B60 = [0]  # -> sagt nichts ueber die relative Lage zu den anderen aus

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


def fit_mask(n, exclude):
    """
    Baut ein boolesches Array der Laenge n: ueberall True, ausser an den
    Positionen aus "exclude" (dort False).
    True  = Punkt wird fuer den Fit verwendet (normale Farbe im Plot)
    False = Punkt wird ausgeschlossen (graue Darstellung im Plot)
    """
    mask = np.ones(n, dtype=bool)
    mask[exclude] = False
    return mask


def model(x, A, b):
    """Fitmodell: Rauschamplitude A/sqrt(x) plus konstanter Untergrund b."""
    return A / np.sqrt(x) + b


def plot_single(x_plot, U, std, mask, x_fit_plot, y_fit, xlabel, filename):
    """
    Zeichnet EINEN Einzelplot (Messpunkte mit Fehlerbalken + Fitkurve) und
    speichert ihn ab. Wird unten zweimal pro Messreihe aufgerufen: einmal mit
    x = 1/sqrt(tau) (ergibt eine Gerade), einmal mit x = tau selbst (ergibt
    die abfallende Kurve, an der man "mehr tau -> weniger Rauschen" direkt
    ablesen kann).

    "mask" ist die boolesche Maske aus fit_mask(): Punkte mit mask=True
    (fuer den Fit verwendet) werden in der normalen Farbe gezeichnet, Punkte
    mit mask=False (ausgeschlossen) grau und durchsichtig.

    Wichtig: diese Funktion transformiert selbst nichts. "x_plot"/"x_fit_plot"
    sind bereits die fertigen Werte fuer die x-Achse (z.B. schon 1/sqrt(tau)
    gerechnet), "y_fit" sind die fertig mit model() berechneten y-Werte der
    Fitkurve. So kann dieselbe Funktion fuer beide Varianten (linear und
    gegen tau) benutzt werden, ohne dass sie wissen muss, welche Transformation
    gerade verwendet wird.
    """
    # Ausgeschlossene Punkte zuerst (im Hintergrund), grau/durchsichtig.
    plt.errorbar(x_plot[~mask], U[~mask] * 10**6, yerr=2 * std[~mask] * 10**6,
                 fmt="o", capsize=3, color="gray", ecolor="gray", alpha=0.4)
    # Fuer den Fit verwendete Punkte, normale Farbe.
    plt.errorbar(x_plot[mask], U[mask] * 10**6, yerr=2 * std[mask] * 10**6,
                 fmt="o", capsize=3, color="#639A00", ecolor="#82C80097")
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

# Masken bauen: legen fest, welche Punkte pro Messreihe in den Fit eingehen.
mask_wi = fit_mask(len(x_wi), EXCLUDE_WI)
mask_oi = fit_mask(len(x_oi), EXCLUDE_OI)
if PLOT_20S:
    mask_b20 = fit_mask(len(x_b20), EXCLUDE_B20)
if PLOT_60S:
    mask_b60 = fit_mask(len(x_b60), EXCLUDE_B60)

# -----------------------------------------------------------------------
# Fits: jede Messreihe bekommt ihren eigenen Fit von model(x, A, b), aber
# NUR auf Basis der Punkte, die durch die jeweilige Maske erlaubt sind
# (x_wi[mask_wi] statt x_wi, usw.).
# curve_fit(model, x, U) gibt ein Tupel (popt, pcov) zurueck:
#   popt = die gefundenen besten Parameter, hier [A, b]
#   pcov = die Kovarianzmatrix des Fits -- die Wurzel der Diagonalelemente
#          ist die Unsicherheit des jeweiligen Parameters:
#          sqrt(pcov[0, 0]) -> Unsicherheit von A, sqrt(pcov[1, 1]) -> von b
# -----------------------------------------------------------------------
popt_wi, pcov_wi = curve_fit(model, x_wi[mask_wi], U_wi[mask_wi])
popt_oi, pcov_oi = curve_fit(model, x_oi[mask_oi], U_oi[mask_oi])
print(f"A_wi = {popt_wi[0]:.3e} +- {np.sqrt(pcov_wi[0, 0]):.3e}")
print(f"A_oi = {popt_oi[0]:.3e} +- {np.sqrt(pcov_oi[0, 0]):.3e}")
print(f"b_wi = {popt_wi[1]:.3e} +- {np.sqrt(pcov_wi[1, 1]):.3e}")
print(f"b_oi = {popt_oi[1]:.3e} +- {np.sqrt(pcov_oi[1, 1]):.3e}")

if PLOT_20S:
    popt_b20, pcov_b20 = curve_fit(model, x_b20[mask_b20], U_b20[mask_b20])
    print(f"A_b20 = {popt_b20[0]:.3e} +- {np.sqrt(pcov_b20[0, 0]):.3e}")
    print(f"b_b20 = {popt_b20[1]:.3e} +- {np.sqrt(pcov_b20[1, 1]):.3e}")

if PLOT_60S:
    popt_b60, pcov_b60 = curve_fit(model, x_b60[mask_b60], U_b60[mask_b60])
    print(f"A_b60 = {popt_b60[0]:.3e} +- {np.sqrt(pcov_b60[0, 0]):.3e}")
    print(f"b_b60 = {popt_b60[1]:.3e} +- {np.sqrt(pcov_b60[1, 1]):.3e}")

# -----------------------------------------------------------------------
# Fuer die Fit-Kurven im Plot brauchen wir viele, dicht liegende x-Werte.
# Wir nehmen bewusst den Bereich ALLER Punkte (auch der ausgeschlossenen),
# damit man im Plot sieht, wie gut/schlecht die Fitkurve auch die grauen
# Punkte trifft -- nicht nur den Bereich, der tatsaechlich gefittet wurde.
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
plot_single(1 / np.sqrt(x_wi), U_wi, std_wi, mask_wi,
            1 / np.sqrt(x_fit_wi), model(x_fit_wi, *popt_wi),
            r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_wi.pdf")
plot_single(x_wi, U_wi, std_wi, mask_wi,
            x_fit_wi, model(x_fit_wi, *popt_wi),
            r"$\tau$ [s]", f"{OUTDIR}/time_constant_wi_vs_tau.pdf")

plot_single(1 / np.sqrt(x_oi), U_oi, std_oi, mask_oi,
            1 / np.sqrt(x_fit_oi), model(x_fit_oi, *popt_oi),
            r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_oi.pdf")
plot_single(x_oi, U_oi, std_oi, mask_oi,
            x_fit_oi, model(x_fit_oi, *popt_oi),
            r"$\tau$ [s]", f"{OUTDIR}/time_constant_oi_vs_tau.pdf")

if PLOT_20S:
    plot_single(1 / np.sqrt(x_b20), U_b20, std_b20, mask_b20,
                1 / np.sqrt(x_fit_b20), model(x_fit_b20, *popt_b20),
                r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_b20.pdf")
    plot_single(x_b20, U_b20, std_b20, mask_b20,
                x_fit_b20, model(x_fit_b20, *popt_b20),
                r"$\tau$ [s]", f"{OUTDIR}/time_constant_b20_vs_tau.pdf")

if PLOT_60S:
    plot_single(1 / np.sqrt(x_b60), U_b60, std_b60, mask_b60,
                1 / np.sqrt(x_fit_b60), model(x_fit_b60, *popt_b60),
                r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_b60.pdf")
    plot_single(x_b60, U_b60, std_b60, mask_b60,
                x_fit_b60, model(x_fit_b60, *popt_b60),
                r"$\tau$ [s]", f"{OUTDIR}/time_constant_b60_vs_tau.pdf")

# -----------------------------------------------------------------------
# Gemeinsamer Plot: alle aktuell eingeschalteten Messreihen zusammen, mit
# je einer eigenen Farbe. Auch hier beide Varianten (1/sqrt(tau) und tau).
# Es werden bewusst nur die Messpunkte (keine Fitkurven) gezeigt, um den
# Plot nicht zu ueberladen. Ausgeschlossene Punkte werden hier NICHT extra
# grau markiert (das ist nur bei den Einzelplots wichtig, wo der Fit direkt
# danebengezeichnet wird) -- alle Punkte jeder Messreihe erscheinen normal.
#
# Zusaetzlich gibt es ein "Zoom-Inset": eine kleine zweite Achse OBEN
# RECHTS im Plot, die denselben Datensatz nochmal zeigt, aber mit engerem
# x-/y-Bereich -- so sieht man trotz des einen Punktes mit riesigem
# Fehlerbalken (der die y-Achse sonst auseinanderzieht) noch, wie nah die
# anderen, kleineren Werte beieinander liegen.
# -----------------------------------------------------------------------
def plot_combined(x_wi_plot, x_oi_plot, x_b20_plot, x_b60_plot, xlabel, filename,
                   zoom_xlim=None, zoom_ylim=None):
    """
    Zeichnet den gemeinsamen Plot fuer eine gegebene x-Achsen-Variante,
    optional inklusive eines gezoomten Insets.

    zoom_xlim / zoom_ylim: Tupel (min, max), legen fest, welcher Ausschnitt
    im Inset zu sehen ist. Das ist der Hebel, an dem man dreht, um den Zoom
    zu veraendern -- siehe die Aufrufe von plot_combined() ganz unten.
    Werden beide auf None gelassen (Standard), wird GAR KEIN Inset gezeichnet
    -- so laesst sich der Zoom pro Aufruf ein-/ausschalten.
    """
    fig, ax = plt.subplots()

    def draw_series(axis, x_plot, U, std, color, ecolor, label=None):
        """
        Zeichnet EINE Messreihe (Fehlerbalkenplot) auf eine gegebene Achse.
        Wird unten zweimal pro Messreihe aufgerufen: einmal mit axis=ax
        (Hauptplot), einmal mit axis=axins (Inset) -- so muss der Plot-Code
        pro Messreihe nur einmal geschrieben werden, statt Hauptplot und
        Inset getrennt mit denselben vier errorbar-Aufrufen zu duplizieren.
        """
        axis.errorbar(x_plot, U * 10**6, yerr=2 * std * 10**6, fmt="o",
                       capsize=3, label=label, color=color, ecolor=ecolor)

    # --------------------------- Hauptplot ---------------------------
    draw_series(ax, x_wi_plot, U_wi, std_wi, "#639A00", "#82C80097", "with Illumination")
    draw_series(ax, x_oi_plot, U_oi, std_oi, "#FF9100", "#FF91009D", "without Illumination")
    if PLOT_20S:
        draw_series(ax, x_b20_plot, U_b20, std_b20, "#3366CC", "#3366CC97", "Batterie, 20s Messung")
    if PLOT_60S:
        draw_series(ax, x_b60_plot, U_b60, std_b60, "#9933CC", "#9933CC97", "Batterie, 60s Messung")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"U [$\mu V$]")
    ax.legend()
    ax.grid()

    # ------------------------- Zoom-Inset -------------------------
    # Nur zeichnen, wenn beim Aufruf tatsaechlich ein Zoom-Bereich angegeben
    # wurde. zoom_xlim/zoom_ylim weglassen (oder =None setzen) -> kein Inset.
    if zoom_xlim is not None and zoom_ylim is not None:
        # ax.inset_axes([x0, y0, breite, hoehe]) erzeugt eine zweite,
        # kleinere Achse INNERHALB der Hauptachse. Alle vier Zahlen sind
        # relativ zur Hauptachse (0 = links/unten, 1 = rechts/oben).
        # ANPASSEN: diese vier Zahlen veraendern, um GROESSE (3./4. Zahl)
        # und POSITION (1./2. Zahl) des Insets zu aendern.
        axins = ax.inset_axes([0.1, 0.25, 0.45, 0.4])

        # Dieselben Messreihen nochmal auf das Inset zeichnen (identischer
        # Code wie oben fuer den Hauptplot, nur auf "axins" statt "ax" --
        # daher die draw_series()-Hilfsfunktion). Ohne "label", da eine
        # zweite Legende im kleinen Inset nur unnoetig Platz wegnehmen wuerde.
        draw_series(axins, x_wi_plot, U_wi, std_wi, "#639A00", "#82C80097")
        draw_series(axins, x_oi_plot, U_oi, std_oi, "#FF9100", "#FF91009D")
        if PLOT_20S:
            draw_series(axins, x_b20_plot, U_b20, std_b20, "#3366CC", "#3366CC97")
        if PLOT_60S:
            draw_series(axins, x_b60_plot, U_b60, std_b60, "#9933CC", "#9933CC97")
        axins.grid()

        # Der eigentliche "Zoom": den sichtbaren Bereich des Insets auf
        # einen kleinen Ausschnitt einschraenken (statt den vollen
        # Datenbereich wie im Hauptplot zu zeigen).
        # ANPASSEN: zoom_xlim/zoom_ylim sind Funktionsargumente -- die
        # tatsaechlichen Werte dafuer stehen unten bei den Aufrufen von
        # plot_combined(). Dort die Zahlen aendern, um den gezeigten
        # Ausschnitt zu verschieben oder zu vergroessern/verkleinern.
        axins.set_xlim(*zoom_xlim)
        axins.set_ylim(*zoom_ylim)

        # Zeichnet automatisch ein Rechteck um den gezoomten Bereich im
        # Hauptplot sowie Verbindungslinien zum Inset -- macht auf einen
        # Blick klar, welcher Ausschnitt da vergroessert dargestellt wird.
        ax.indicate_inset_zoom(axins, edgecolor="black")

    fig.savefig(filename)
    plt.close(fig)


# Fuer die 1/sqrt(tau)-Variante liegen die kleinen, eng beieinanderliegenden
# Werte bei kleinem x (siehe Hauptplot) -- daher wird dort reingezoomt.
plot_combined(
    1 / np.sqrt(x_wi), 1 / np.sqrt(x_oi),
    1 / np.sqrt(x_b20) if PLOT_20S else None,
    1 / np.sqrt(x_b60) if PLOT_60S else None,
    r"$1/\sqrt{\tau}$ [1/s]", f"{OUTDIR}/time_constant_combined.pdf",
    zoom_xlim=(0.5, 2), zoom_ylim=(0, 0.3),
)
# Bei der tau-Variante kein Zoom-Inset (zoom_xlim/zoom_ylim weggelassen).
plot_combined(
    x_wi, x_oi,
    x_b20 if PLOT_20S else None,
    x_b60 if PLOT_60S else None,
    r"$\tau$ [s]", f"{OUTDIR}/time_constant_combined_vs_tau.pdf",
)
