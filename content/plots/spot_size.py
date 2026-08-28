import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------
# Leistung des Lasers am Ort der Messung, in mW (gleiche Einheit wie die
# geladenen Leistungsmesswerte). Hier von Hand eintragen, wird ganz unten
# fuer die Intensitaetsberechnung (Leistung pro Flaeche) benutzt.
# -----------------------------------------------------------------------
LASER_POWER = 34.5  # mW

# -----------------------------------------------------------------------
# Maske: nur Messpunkte, deren Position (in mm) in EINEM dieser Bereiche
# liegt, werden fuer den jeweiligen Gauss-Fit benutzt. Alle anderen Punkte
# werden im Plot trotzdem gezeigt (grau/durchsichtig), aber nicht mitgefittet.
# Jede Zeile ist ein (min, max)-Tupel; es koennen beliebig viele Bereiche
# angegeben werden (z.B. um einen Ausreisser mitten im Peak auszusparen) --
# einfach ein weiteres (min, max)-Tupel in die Liste einfuegen.
# -----------------------------------------------------------------------
# X_FIT_RANGES = [(13, 19),(8,9.5)]
# Y_FIT_RANGES = [(8, 18.5)]

X_FIT_RANGES = [(7,20)]
Y_FIT_RANGES = [(8, 22)]

x, Ax = np.genfromtxt("data/spot_size/spot_size_x.txt", unpack=True)
y, Ay = np.genfromtxt("data/spot_size/spot_size_y.txt", unpack=True)
x = x * 10**-3  # Rohdaten sind in Mikrometern, wir wollen Millimeter
y = y * 10**-3

# Ableitung der Knife-Edge-Kurve (Leistung ueber Position) ist naeherungsweise
# das eigentliche Strahlprofil (Gauss-foermig), siehe fruehere Erklaerung.
# Bei der y-Messung faellt die Leistung mit steigender Position -- deshalb
# das Minus, damit hier auch ein positiver Peak entsteht statt eines Tals.
# dif_x = np.diff(Ax)
# dif_y = -np.diff(Ay)

dif_x = np.gradient(Ax,x)
dif_y = -np.gradient(Ay,y)

# Jeder Differenzwert liegt "zwischen" zwei benachbarten Messpunkten; wir
# nehmen als zugehoerige Position einfach den jeweils ersten der beiden
# Punkte (x[:-1] hat deshalb genau die gleiche Laenge wie dif_x).
# x_diff = x[:-1]
# y_diff = y[:-1]
x_diff = x
y_diff = y


def gauss(pos, A, pos0, sigma, offset):
    """Gauss-Kurve: Amplitude A, Zentrum pos0, Breite sigma, Untergrund offset."""
    return A * np.exp(-(pos - pos0) ** 2 / (2 * sigma**2)) + offset


def fit_gauss_masked(pos, values, fit_ranges):
    """
    Fittet eine Gausskurve nur an die Punkte, deren Position in EINEM der
    Bereiche aus "fit_ranges" liegt (die "Maske"). "fit_ranges" ist eine
    Liste von (min, max)-Tupeln -- ein Punkt wird verwendet, sobald er in
    mindestens einem dieser Bereiche liegt (Vereinigung/ODER-Verknuepfung
    aller Bereiche, kein UND).

    Rueckgabe:
      mask - boolesches Array (gleiche Laenge wie pos), True = Punkt wurde
             fuer den Fit benutzt
      popt - gefundene Parameter [A, pos0, sigma, offset]
      pcov - Kovarianzmatrix des Fits (Wurzel der Diagonale = Unsicherheiten)
    """
    # Maske Bereich fuer Bereich aufbauen: erst ueberall False, dann fuer
    # jeden (min, max)-Bereich die passenden Punkte per "|=" (logisches ODER,
    # in-place) dazuschalten.
    mask = np.zeros(len(pos), dtype=bool)
    for fit_min, fit_max in fit_ranges:
        mask |= (pos >= fit_min) & (pos <= fit_max)

    # Startwerte fuer curve_fit, damit der Fit zuverlaessig konvergiert:
    # Amplitude = hoechster Wert im (gesamten) Fit-Bereich, Zentrum = dessen
    # Position, Breite = grobe Schaetzung als ein Viertel der Spannweite
    # aller verwendeten Punkte, Untergrund = 0 (die Differenzkurve sollte im
    # Mittel um 0 schwanken).
    p0 = [
        values[mask].max(),
        pos[mask][values[mask].argmax()],
        (pos[mask].max() - pos[mask].min()) / 4,
        0.0,
    ]
    popt, pcov = curve_fit(gauss, pos[mask], values[mask], p0=p0)
    return mask, popt, pcov


mask_x, popt_x, pcov_x = fit_gauss_masked(x_diff, dif_x, X_FIT_RANGES)
mask_y, popt_y, pcov_y = fit_gauss_masked(y_diff, dif_y, Y_FIT_RANGES)

err_x = np.sqrt(np.diag(pcov_x))
err_y = np.sqrt(np.diag(pcov_y))
print(f"x: A={popt_x[0]:.3f}+-{err_x[0]:.3f}, x0={popt_x[1]:.3f}+-{err_x[1]:.3f} mm, "
      f"sigma={popt_x[2]:.3f}+-{err_x[2]:.3f} mm, offset={popt_x[3]:.3f}+-{err_x[3]:.3f}")
print(f"y: A={popt_y[0]:.3f}+-{err_y[0]:.3f}, y0={popt_y[1]:.3f}+-{err_y[1]:.3f} mm, "
      f"sigma={popt_y[2]:.3f}+-{err_y[2]:.3f} mm, offset={popt_y[3]:.3f}+-{err_y[3]:.3f}")


def plot_masked_fit(pos, values, mask, popt, xlabel, filename):
    """
    Zeichnet die Differenzkurve: fuer den Fit benutzte Punkte (mask=True)
    farbig, nicht benutzte Punkte grau/durchsichtig -- zusammen mit der
    gefitteten Gausskurve (nur ueber den Bereich der benutzten Punkte).
    """
    fig, ax = plt.subplots(layout="constrained")

    # ax.plot(pos[~mask], values[~mask], "x", color="gray", alpha=0.5,
            #  label="nicht im Fit verwendet")
    ax.plot(pos[mask], values[mask], "o", color="#639A00",
             label="Messdaten")

    pos_fit = np.linspace(pos[mask].min(), pos[mask].max(), 500)
    if filename == "pdf/spot_size_x.pdf":
        pos_fit = np.linspace(8, pos[mask].max(), 500)



    ax.plot(pos_fit, gauss(pos_fit, *popt), "-", color="#FF9100", label="Gauss-Fit")

    ax.set_xlabel(xlabel)
    if xlabel == "Entfernung x / mm":
        ax.set_ylabel(r"$\frac{\text{d} P}{\text{d} x}$ / $\frac{\text{mW}}{\text{mm}}$")
    else:
        ax.set_ylabel(r"$\frac{\text{d} P}{\text{d} y}$ / $\frac{\text{mW}}{\text{mm}}$")
    
    ax.legend()
    ax.grid()
    fig.savefig(filename)
    plt.close(fig)


plot_masked_fit(x_diff, dif_x, mask_x, popt_x, "Entfernung x / mm", "pdf/spot_size_x.pdf")
plot_masked_fit(y_diff, dif_y, mask_y, popt_y, "Entfernung y / mm", "pdf/spot_size_y.pdf")

# -----------------------------------------------------------------------
# Strahlbreite w = 2*sigma pro Richtung (Definition des Strahlradius bei
# 1/e^2 der Maximalintensitaet, siehe fruehere Erklaerung zur Spotgroesse).
# Daraus die effektive Flaeche des (elliptischen) Strahlquerschnitts
# A = pi * w_x * w_y und schliesslich die Intensitaet I = P / A der oben
# eingegebenen Laserleistung LASER_POWER.
#
# Unsicherheiten werden per Fehlerfortpflanzung durchgereicht:
#   - w = 2*sigma -> Fehler von w ist einfach 2x Fehler von sigma
#     (linearer Zusammenhang, der Faktor 2 wird direkt mit durchmultipliziert)
#   - A = pi*w_x*w_y ist ein Produkt -> die RELATIVEN Fehler von w_x und w_y
#     werden quadratisch addiert (Standardformel fuer Produkte/Quotienten):
#     (dA/A)^2 = (dw_x/w_x)^2 + (dw_y/w_y)^2
#   - I = P/A -> LASER_POWER wird als exakt angenommen (kein eigener
#     Fehler eingegeben), daher hat I denselben relativen Fehler wie A.
# -----------------------------------------------------------------------
w_x = 2 * popt_x[2]
w_y = 2 * popt_y[2]
err_w_x = 2 * err_x[2]
err_w_y = 2 * err_y[2]

A_eff = np.pi * w_x * w_y
err_A_eff = A_eff * np.sqrt((err_w_x / w_x) ** 2 + (err_w_y / w_y) ** 2)

intensity = LASER_POWER / A_eff
err_intensity = intensity * (err_A_eff / A_eff)

print(f"w_x = {w_x:.3f}+-{err_w_x:.3f} mm")
print(f"w_y = {w_y:.3f}+-{err_w_y:.3f} mm")
print(f"A_eff = {A_eff:.3f}+-{err_A_eff:.3f} mm^2")
print(f"Intensitaet = {intensity:.3f}+-{err_intensity:.3f} mW/mm^2")
