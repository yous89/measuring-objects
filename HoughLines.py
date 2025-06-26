# ---------------------------------------------
# Nötige Bibliotheken importieren
# ---------------------------------------------
import cv2
import numpy as np
from imutils import perspective
import os

from numpy import dtype


# ---------------------------------------------
# Bild einlesen und skalieren
# ---------------------------------------------
def load_and_resize(image_path, scale=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild '{image_path}' konnte nicht geladen werden.")
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)))

# ---------------------------------------------
# Kantenverarbeitung: Blur, Canny, Dilate
# ---------------------------------------------
def processed_edges(image, blur_kernel=(3, 3), canny1=50, canny2=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             # Graustufen
    blurred = cv2.GaussianBlur(gray, blur_kernel, 2)    # Glättung
    edged = cv2.Canny(blurred, canny1, canny2)                 # Canny-Kanten
    dilated = cv2.dilate(edged, None, iterations=2)      # Erweiterung der Kanten
    return dilated

# ---------------------------------------------
# Kontur von A4-Blatt erkennen und entzerren (warp)
# ---------------------------------------------
def find_a4_and_warp(image, epsilon=20):
    processed = processed_edges(image)      # Vorverarbeitung dees Bildes
    save_steps({"processed_edges_for_a4": processed})          # Debug-Speicherung
    cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL,   # Nur äußere Konturen
                               cv2.CHAIN_APPROX_SIMPLE)        # 4 Punkte
    if not cnts:
        raise Exception("Keine Konturen gefunden")
    largest = max(cnts, key=cv2.contourArea)          # größte Kontur nach Fläche
    polygon = cv2.approxPolyDP(largest, epsilon=epsilon, closed=True)      # vereinfachen zu 4 Ecken
    if len(polygon) != 4:
        raise Exception("A4-Kontur hat nicht genau 4 Ecken")
    box = perspective.order_points(polygon.reshape(4, 2))      # sortierte Ecken
    # Zeichne Kontur als geschlossene grüne Linie
    contour_img = image.copy()
    cv2.polylines(contour_img, [np.int32(box)], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, pt in enumerate(box):
        pt = tuple(int(v) for v in pt)
        cv2.circle(contour_img, pt, 6, (255, 255, 0), -1)
        cv2.putText(contour_img, f"{i+1}", (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    save_steps({f"A4_Kontur_und_Ecken": contour_img})
    warped = perspective.four_point_transform(image, box)      # perspektivisch entzerren
    return warped

# ---------------------------------------------
# Winkelberechnung zwischen zwei Linien
# ---------------------------------------------
def angle_between_lines(line1, line2):
    def vec(l): return np.array([l[2] - l[0], l[3] - l[1]])
    v1, v2 = vec(line1), vec(line2)
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0
    angle = np.arccos(np.clip(dot / norms, -1.0, 1.0))
    return np.degrees(angle)

# ---------------------------------------------
# Schnittpunkt zweier Linien berechnen
# ---------------------------------------------
def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Parallel oder identisch
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

# ---------------------------------------------
# Beschriftet die gemessene Länge in mm + zeigt Fehler an
# ---------------------------------------------
def compute_relative_error(measured_length_mm, measured_width_mm, true_length_mm, true_width_mm):
    error1 = abs(measured_length_mm - true_length_mm) / true_length_mm * 100
    error2 = abs(measured_width_mm - true_width_mm) / true_width_mm * 100
    return error1, error2

# ---------------------------------------------
# Finde das beste orthogonale Linienpaar (L-Form) und messe deren Länge
# ---------------------------------------------
def apply_hough_lines(image, edge_image, pixels_per_mm, true_length_mm, true_width_mm):
    img = image.copy()
    lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)
    if lines is None:
        return img
    debug_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Grün: alle Linien
        save_steps({ "debug_all_hough_lines": debug_img })
    best_pair = None
    best_score = 0
    lengths = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l1 = lines[i][0]
            l2 = lines[j][0]
            angle = angle_between_lines(l1, l2) # WWinkel berechnen
            if 85 <= angle <= 95: # annähernd rechten Winkel prüfen
                # Länge des Paares berechnen
                len1 = np.hypot(l1[2] - l1[0], l1[3] - l1[1])
                len2 = np.hypot(l2[2] - l2[0], l2[3] - l2[1])
                total_length = len1 + len2
                # Das Paar mit der größten Gesamtlänge, das einen rechten Winkel
                if total_length > best_score:
                    best_score = total_length
                    best_pair = (l1, l2)
    if best_pair:
        l1, l2 = best_pair
        # Das Paar zeichnen
        cv2.line(img, (l1[0], l1[1]), (l1[2], l1[3]), (0, 255, 0), 2) # grün
        cv2.line(img, (l2[0], l2[1]), (l2[2], l2[3]), (0, 255, 0), 2) # grün
        inter_pt = compute_intersection(l1, l2) # Schnittpunkt berechnen
        if inter_pt:
            cv2.circle(img, inter_pt, 6, (0, 0, 255), -1) # rot
            # Länge ab Schnittpunkt messen
            for line in [l1, l2]:
                d1 = np.linalg.norm(np.array([line[0], line[1]]) - np.array(inter_pt)) # Abstand(Schnittpunkt, Punkt A)
                d2 = np.linalg.norm(np.array([line[2], line[3]]) - np.array(inter_pt)) # Abstand(Schnittpunkt, Punkt B)
                dist_px = max(d1, d2) # echte Kantenlänge (bis zum entfernteren Ende)
                # Von Pixel in mm umrechnen
                dist_mm = dist_px / pixels_per_mm
                lengths.append(dist_mm)
    if len(lengths) == 2:
        # Sortieren: [länge_mm, breite_mm]
        length_mm, width_mm = sorted(lengths, reverse=True)
        # Relative Fehler berechnen
        error1 , error2 = compute_relative_error(length_mm, width_mm, true_length_mm, true_width_mm)
        # Textanzeige unten links
        h = img.shape[0]
        lines_text = [
            f"Lange Kante: {length_mm:.1f} mm (rel. Fehler: {error1:.1f} %)",
            f"Kurze Kante: {width_mm:.1f} mm (rel. Fehler: {error2:.1f} %)"
        ]
        for i, line in enumerate(lines_text[::-1]):
            y = h - 10 - i * 22
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
    return img

# ---------------------------------------------
# Speichert Zwischenergebnisse
# ---------------------------------------------
def save_steps(steps, output_dir="output_steps"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name, img in steps.items():
        path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(path, img)
        print(f"Gespeichert: {path}")

# ---------------------------------------------
# Hauptpipeline: Verarbeitung & Ausmessung
# ---------------------------------------------
def run(image_path, name = ""):
    img = load_and_resize(image_path)                           # Lade und skaliere Bild
    warped = find_a4_and_warp(img)                              # A4 finden und entzerren
    edges = processed_edges(warped,                      # Objektkanten extrahieren
                            blur_kernel=(7, 7))          # unterdrückt Rauschen
    ppm = (warped.shape[1] / 210 + warped.shape[0] / 297) / 2   # Pixel/mm basierend auf A4
    result = apply_hough_lines(warped.copy(), edges, ppm,
                    true_length_mm=120.5, true_width_mm=29.0)   # L-Messung auf Objekt
                    # true_length_mm=86.0, true_width_mm=54.0)   # L-Messung auf Test-Objekt (Ausweis)
    # Ergebnisse abspeichern
    save_steps({
        f"{name}_Warped_A4": warped,
        f"{name}_Edges": edges,
        f"{name}_HoughLines_Measurement": result
    })

# ---------------------------------------------
# Starte das Skript
# ---------------------------------------------
if __name__ == "__main__":
    run("images/Inbus_rotated3.jpg", name="Inbus_rotated3")  # Pfad ggf. anpassen
