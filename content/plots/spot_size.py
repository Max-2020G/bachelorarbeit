import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------
# Maske: nur Messpunkte, deren Position (in mm) in diesem Bereich liegt,
# werden fuer den jeweiligen Gauss-Fit benutzt. Alle anderen Punkte werden
# im Plot trotzdem gezeigt (grau/durchsichtig), aber nicht mitgefittet.
# Bereiche hier von Hand anpassen, je nachdem welcher Ausschnitt der Kurve
# das eigentliche Strahlprofil ist und welcher nur Rauschen.
# -----------------------------------------------------------------------
X_FIT_MIN, X_FIT_MAX = 13, 17.5
Y_FIT_MIN, Y_FIT_MAX = 8, 18.5

x, Ax = np.genfromtxt("data/spot_size/spot_size_x.txt", unpack=True)
y, Ay = np.genfromtxt("data/spot_size/spot_size_y.txt", unpack=True)
x = x * 10**-3  # Rohdaten sind in Mikrometern, wir wollen Millimeter
y = y * 10**-3

# Ableitung der Knife-Edge-Kurve (Leistung ueber Position) ist naeherungsweise
# das eigentliche Strahlprofil (Gauss-foermig), siehe fruehere Erklaerung.
# Bei der y-Messung faellt die Leistung mit steigender Position -- deshalb
# das Minus, damit hier auch ein positiver Peak entsteht statt eines Tals.
dif_x = np.diff(Ax)
dif_y = -np.diff(Ay)

# Jeder Differenzwert liegt "zwischen" zwei benachbarten Messpunkten; wir
# nehmen als zugehoerige Position einfach den jeweils ersten der beiden
# Punkte (x[:-1] hat deshalb genau die gleiche Laenge wie dif_x).
x_diff = x[:-1]
y_diff = y[:-1]


def gauss(pos, A, pos0, sigma, offset):
    """Gauss-Kurve: Amplitude A, Zentrum pos0, Breite sigma, Untergrund offset."""
    return A * np.exp(-(pos - pos0) ** 2 / (2 * sigma**2)) + offset


def fit_gauss_masked(pos, values, fit_min, fit_max):
    """
    Fittet eine Gausskurve nur an die Punkte, deren Position zwischen
    fit_min und fit_max liegt (die "Maske").

    Rueckgabe:
      mask - boolesches Array (gleiche Laenge wie pos), True = Punkt wurde
             fuer den Fit benutzt
      popt - gefundene Parameter [A, pos0, sigma, offset]
      pcov - Kovarianzmatrix des Fits (Wurzel der Diagonale = Unsicherheiten)
    """
    mask = (pos >= fit_min) & (pos <= fit_max)

    # Startwerte fuer curve_fit, damit der Fit zuverlaessig konvergiert:
    # Amplitude = hoechster Wert im Fit-Bereich, Zentrum = dessen Position,
    # Breite = grobe Schaetzung als ein Viertel der Bereichsbreite,
    # Untergrund = 0 (die Differenzkurve sollte im Mittel um 0 schwanken).
    p0 = [
        values[mask].max(),
        pos[mask][values[mask].argmax()],
        (fit_max - fit_min) / 4,
        0.0,
    ]
    popt, pcov = curve_fit(gauss, pos[mask], values[mask], p0=p0)
    return mask, popt, pcov


mask_x, popt_x, pcov_x = fit_gauss_masked(x_diff, dif_x, X_FIT_MIN, X_FIT_MAX)
mask_y, popt_y, pcov_y = fit_gauss_masked(y_diff, dif_y, Y_FIT_MIN, Y_FIT_MAX)

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

    ax.plot(pos[~mask], values[~mask], "x", color="gray", alpha=0.5,
             label="nicht im Fit verwendet")
    ax.plot(pos[mask], values[mask], "o", color="#639A00",
             label="im Fit verwendet")

    pos_fit = np.linspace(pos[mask].min(), pos[mask].max(), 500)
    ax.plot(pos_fit, gauss(pos_fit, *popt), "-", color="#FF9100", label="Gauss-Fit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta$Leistung / mW")
    ax.legend()
    ax.grid()
    fig.savefig(filename)
    plt.close(fig)


plot_masked_fit(x_diff, dif_x, mask_x, popt_x, "Entfernung x / mm", "pdf/spot_size_x.pdf")
plot_masked_fit(y_diff, dif_y, mask_y, popt_y, "Entfernung y / mm", "pdf/spot_size_y.pdf")
