# Automatische Objektvermessung mit OpenCV

Dieses Projekt hat zum Ziel, metallische Werkstücke (z.B. Inbusschlüssel) automatisiert anhand eines Kamerabilds auszumessen.

## Installation & Setup

1. **Virtuelle Umgebung erstellen**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Benötigte Bibliotheken installieren**
   ```bash
   pip install opencv-python matplotlib numpy
   ```

## Problemstellung

In modernen Fräsmaschinen kann ein falsch dimensionierter Rohling zur Kollision mit der Spindel führen – dies verursacht teure Schäden. Ziel ist es daher, eine automatische „**Smarte Aufspannkontrolle**“ zu entwickeln, die:

- Werkstücke auf Bildern erkennt,
- deren Kantenlängen misst,
- und das Ergebnis in **Millimeter** ausgibt.

## Vorgehensweise
![Konzept Final](https://github.com/user-attachments/assets/d1a2f0a2-9e6f-4fbf-8b79-67523e60bac9)

Zur präzisen Vermessung wird folgende Bildverarbeitungskette eingesetzt:

1. **Referenz erkennen**
   Muss unten rechts liegen (z.B. Ausweis, Münze, usw.). Dient als metrische Referenz. Es wird über Kantenerkennung und Konturenlokalisierung im Bild erkannt.

2. **Objekterkennung per Konturanalyse**
   Im entzerrten Bild werden relevante Objekte erkannt, Randbereiche ignoriert, und mit **Bounding Boxes** versehen.

3**Umrechnung von Pixel in Millimeter**
   Da die reale Größe eines gegebenen Referenz bekannt ist (Ausweis: z.B. 85,6 mm x 50,4 mm), wird die Auflösung im Bild berechnet, durch die Breite des Referenz, nähmlich hier nur 85,6 muss als Parameter gegeben wird:

   ```python
   pixels_per_metric = ref_length_px / known_width_mm
   ```

   Mit dieser Auflösung werden erkannte Objektkanten umgerechnet:

   ```python
     # Umrechnung von Pixel zu mm
     width_mm = width_px / pixels_per_metric
     height_mm = height_px / pixels_per_metric
   ```

   Die gemessenen Werte werden direkt im Bild als **Längenbeschriftung** eingeblendet.

## Ergebnis
![BoundingBox_Measurement](https://github.com/user-attachments/assets/d84ab600-3fc2-49d3-b350-2972f350f674)


## Besonderheiten

- Die größte Herausforderung besteht in der präzisen Vermessung von asymmetrischen Objekten wie einem Inbusschlüssel.
- Lichtverhältnisse und Bildschärfe haben entscheidenden Einfluss auf die Genauigkeit der Ergebnisse.
- Erweiterbar mit Kalibrierung über bekannte Objekte (z.B. Münze, Papierkante).

## Referenzen
- [Contour Detektion](https://learnopencv.com/contour-detection-using-opencv-python-c/)
- [Measuring Size - Bounding Box](https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/)
- [Contour Approximation](https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/)
- [Grafik zeichnen](https://excalidraw.com/)
