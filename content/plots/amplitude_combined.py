import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

A_ZnTe, U_ZnTe = np.genfromtxt("data/amplitude/amplitudes_ZnTe_right_power.txt", unpack=True)
A_OH1, U_OH1 = np.genfromtxt("data/amplitude/amplitudes_right_power.txt", unpack=True)
U_ZnTe = U_ZnTe * 10**6
U_OH1 = U_OH1 * 10**6
Fl = 2.241 * 10**-2  # in cm^2
Faktor = A_OH1 / 34.5
A_OH1 = 0.103 * Faktor * Fl


def lin(x, m, b):
    return m * x + b


# Fits mit eigenen Variablennamen pro Messreihe (statt popt/m/b/x wie
# vorher wiederzuverwenden) -- sonst ueberschreibt der OH1-Fit die
# ZnTe-Fitparameter, und man kann die ZnTe-Fitkurve spaeter (z.B. im
# Zoom-Inset unten) nicht mehr zeichnen.
popt_znte, pcov_znte = curve_fit(lin, A_ZnTe, U_ZnTe)
x_znte = np.linspace(A_ZnTe.min(), A_ZnTe.max(), 1000)

popt_oh1, pcov_oh1 = curve_fit(lin, A_OH1, U_OH1)
x_oh1 = np.linspace(A_OH1.min(), A_OH1.max(), 1000)

fig, ax = plt.subplots(layout="constrained")
ax.plot(A_ZnTe, U_ZnTe, "x", color="#639A00")
ax.plot(x_znte, lin(x_znte, *popt_znte), "-", color="#FF9100")
ax.plot(A_OH1, U_OH1, "x", color="#639A00")
ax.plot(x_oh1, lin(x_oh1, *popt_oh1), "-", color="#FF9100")

# ------------------------- Zoom-Inset -------------------------
# axins ist eine zweite, kleinere Achse INNERHALB der Hauptachse.
# [x0, y0, breite, hoehe] sind relativ zur Hauptachse (0 bis 1), hier:
# 40% breit, 35% hoch, oben rechts platziert. Vorher standen hier
# [0, 0.6, 15, 80] -- Breite/Hoehe (15, 80) mussten aber zwischen 0 und 1
# liegen, ausserdem wurde der Rueckgabewert nie in einer Variable
# gespeichert, wodurch nie Daten ins Inset gezeichnet wurden.
axins = ax.inset_axes([0.55, 0.1, 0.4, 0.35], xlim=(-0.01, 0.06), ylim=(-5, 120))
axins.plot(A_ZnTe, U_ZnTe, "x", color="#639A00")
axins.plot(x_znte, lin(x_znte, *popt_znte), "-", color="#FF9100")
axins.plot(A_OH1, U_OH1, "x", color="#639A00")
axins.plot(x_oh1, lin(x_oh1, *popt_oh1), "-", color="#FF9100")
axins.grid()

# Zeichnet automatisch ein Rechteck um den gezoomten Bereich im Hauptplot
# sowie Verbindungslinien zum Inset.
ax.indicate_inset_zoom(axins, edgecolor="black")

ax.set_xlabel("Power / mW")
ax.set_ylabel("Voltage / mikroV")

ax.grid()
fig.savefig("pdf/amplitude_comb.pdf")
