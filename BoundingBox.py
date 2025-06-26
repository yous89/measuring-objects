import cv2
import numpy as np
import os
from imutils import perspective

# ---------- Hilfsfunktionen ----------
# Bild laden und Fehler werfen, falls das Bild nicht gefunden wird
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild '{image_path}' konnte nicht geladen werden.")
    return image
# Sicherstellen, dass ein Verzeichnis existiert, andernfalls wird es erstellt
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
# Bild unter einem bestimmten Namen speichern
def save_image(image, name, output_dir="output_steps"):
    ensure_dir(output_dir)
    cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), image)
# Vorverarbeitung: Graustufen, Weichzeichnen, Kanten finden, Kanten verbessern
def preprocess(image, step_name=""):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, f"{step_name}_1_gray")
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    save_image(blurred, f"{step_name}_2_blurred")
    edged = cv2.Canny(blurred, 50, 150)
    save_image(edged, f"{step_name}_3_edged")
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    save_image(edged, f"{step_name}_4_processed_edges")
    return edged
# Hilfsfunktion zur Berechnung des Mittelpunktes zweier Punkte
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# Hilfsfunktion zum Rechteck-Analyse
def get_box_and_dimensions(contour):
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    box = np.array(box, dtype="int")
    (tl, tr, br, bl) = perspective.order_points(box)
    # Kantenlängen berechnen
    edge1 = np.linalg.norm(tl - tr)
    edge2 = np.linalg.norm(tr - br)
    width_px = max(edge1, edge2)
    height_px = min(edge1, edge2)
    # Für spätere Positionierung
    mid_w = midpoint(tl, tr) if edge1 > edge2 else midpoint(tr, br)
    mid_h = midpoint(tr, br) if edge1 > edge2 else midpoint(tl, tr)
    return box, width_px, height_px, mid_w, mid_h

# ---------- Referenzmessung ----------
# Bestimmt den Umrechnungsfaktor von Pixel zu mm anhand eines bekannten Objekts
def find_reference_object(image, edged, known_width_mm):
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        raise Exception("Keine Konturen gefunden.")
    # Wähle die Kontur mit Mittelpunkt am weitesten unten rechts
    reference_cnt = max(
        cnts,
        key=lambda c: (
                cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2 +
                cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2
        )
    )
    # Maße berechnen
    box, width_px, height_px, _, _ = get_box_and_dimensions(reference_cnt)
    cv2.putText(image, "Referenz: ", (box[0][0], box[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    ref_length_px = max(width_px, height_px)
    pixels_per_metric = ref_length_px / known_width_mm
    return pixels_per_metric

# ---------- Messung der restlichen Objekte ----------
def measure_objects(image, edged, pixels_per_metric):
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    annotated_image = image.copy()
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue  # Kleine Artefakte überspringen
        # Rechteck und Maße in Pixeln berechnen
        box, width_px, height_px, mid_w, mid_h = get_box_and_dimensions(c)
        # Umrechnung von Pixel zu mm
        width_mm = width_px / pixels_per_metric
        height_mm = height_px / pixels_per_metric
        # Rechteck zeichnen
        cv2.drawContours(annotated_image, [box.astype("int")], -1, (0, 255, 0), 2)
        # Maße als Text ausgeben
        cv2.putText(annotated_image, f"{width_mm:.1f} mm", (int(mid_w[0]) + 10, int(mid_w[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated_image, f"{height_mm:.1f} mm", (int(mid_h[0]) + 10, int(mid_h[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Bild mit Annotation speichern
    save_image(annotated_image, "BoundingBox_Measurement")

# ---------- Hauptablauf ----------
# Führt das gesamte Programm aus
def run(image_path, known_width_mm):
    image = load_image(image_path)
    edged = preprocess(image, "object_detection")
    # Umrechnungsfaktor bestimmen anhand der Referenz
    pixels_per_metric = find_reference_object(image, edged, known_width_mm)
    # Alle Objekte messen und visualisieren
    measure_objects(image, edged, pixels_per_metric)
    print("Messung abgeschlossen. Ergebnisse unter 'output_steps/' gespeichert.")
if __name__ == '__main__':
    run("images/Inbusschluessel.jpg", known_width_mm=85.6)  # Beispiel: Referenz ist ein Ausweis (ID-1 Format)
