"""
Bestimmung des Flächenanteils, den ein Kristall in einer kreisrunden
Halterungsöffnung ("Loch") einnimmt.

Technik:
1. Zuschneiden auf den interessierenden Bereich (Loch + Kristall).
2. Referenzkreis (Loch-Rand) per Hough-Circle-Transform oder manuell finden.
3. Kristall per Farbschwelle (HSV: Sättigung + Helligkeit) vom
   Hintergrund/offenen Loch trennen.
4. Flächenanteil = Kristall-Pixel / Loch-Pixel (innerhalb des Referenzkreises).

Wichtig: Automatische Kreiserkennung (HoughCircles) funktioniert nur
zuverlässig auf einem nahen, kontrastreichen Ausschnitt - nicht auf dem
ganzen Foto (zu viel Hintergrund-Textur -> falsche Kreise). Deshalb immer
erst eng zuschneiden.

Ergebnis IMMER visuell verifizieren (siehe mask_vis.jpg) und bei Bedarf
- die (x,y,r) Werte des Referenzkreises manuell anpassen
- die HSV-Schwellwerte (S_MIN, V_MIN) an die eigene Beleuchtung anpassen
"""

import cv2
import numpy as np

# ---- Einstellungen ----
IMAGE_PATH = "kristall_größe.jpeg"          # enger Zuschnitt um Loch + Kristall
S_MIN, V_MIN = 140, 140          # HSV-Schwelle: Kristall = gesättigt UND hell
# Referenzkreis (x, y, r) in Pixel. None = automatische Suche per Hough.
CIRCLE = None
# Bekannter physikalischer Lochdurchmesser in mm (falls bekannt -> absolute
# Fläche statt nur relativer Anteil). Sonst None.
HOLE_DIAMETER_MM = 2.5

# ---- 1. Bild laden ----
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- 2. Referenzkreis (Loch-Rand) ----
if CIRCLE is None:
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=1000,
        param1=60, param2=25,
        minRadius=int(0.1 * min(img.shape[:2])),
        maxRadius=int(0.5 * min(img.shape[:2])),
    )
    if circles is None:
        raise RuntimeError(
            "Kein Kreis gefunden - CIRCLE manuell setzen, z.B. per Klick-Tool "
            "(matplotlib ginput) auf 3 Randpunkten + Kreis-Fit."
        )
    x, y, r = circles[0, 0]
else:
    x, y, r = CIRCLE
x, y, r = int(round(x)), int(round(y)), int(round(r))
print(f"Referenzkreis: Mitte=({x},{y}) Radius={r}px")

circle_mask = np.zeros(gray.shape, np.uint8)
cv2.circle(circle_mask, (x, y), r, 255, -1)
circle_bool = circle_mask > 0

# ---- 3. Kristall-Segmentierung ----
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
S, V = hsv[:, :, 1], hsv[:, :, 2]
crystal_mask = (S > S_MIN) & (V > V_MIN) & circle_bool

# ---- 4. Flächenanteil ----
hole_px = circle_bool.sum()
crystal_px = crystal_mask.sum()
frac = crystal_px / hole_px
print(f"Loch-Fläche: {hole_px} px")
print(f"Kristall-Fläche: {crystal_px} px")
print(f"Flächenanteil Kristall/Loch: {frac*100:.1f} %")

if HOLE_DIAMETER_MM:
    mm_per_px = HOLE_DIAMETER_MM / (2 * r)
    print(f"Skalierung: {mm_per_px:.4f} mm/px")
    print(f"Kristallfläche absolut: {crystal_px * mm_per_px**2:.3f} mm^2")

# ---- 5. Visualisierung zur Kontrolle ----
overlay = img.copy()
overlay[crystal_mask] = (0, 255, 0)
vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
cv2.circle(vis, (x, y), r, (255, 0, 0), 2)
cv2.imwrite("mask_vis.jpg", vis)
print("Kontrollbild gespeichert: mask_vis.jpg")