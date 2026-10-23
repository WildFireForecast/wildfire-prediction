# WildFireForecast

## Datenquellen und Datenherkunft

### 1. Hauptdatensatz: Mesogeos

Für das Projekt **WildFireForecast** verwenden wir den wissenschaftlichen Datensatz *Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean*.

Der Datensatz enthält historische Waldbrandereignisse sowie meteorologische, geografische und umweltbezogene Informationen aus dem Mittelmeerraum.

**Offizielle Quellen:**

- Projektwebsite: https://orionlab.space.noa.gr/mesogeos/
- GitHub-Repository: https://github.com/Orion-AI-Lab/mesogeos
- Wissenschaftliche Veröffentlichung: https://doi.org/10.48550/arXiv.2306.05144
- Datensatz (Zenodo): https://doi.org/10.5281/zenodo.7473331

**Eigenschaften des Mesogeos-Datensatzes:**

- Geografische Abdeckung: Mittelmeerraum
- Ursprünglicher Zeitraum: 2006–2022
- Räumliche Auflösung: 1 km × 1 km
- Zeitliche Auflösung: täglich
- Enthaltene Informationen: Wetter, Vegetation, Bodenfeuchtigkeit, Topografie, Landbedeckung, menschliche Einflüsse und historische Waldbrände

Die Rohdaten unseres Projekts liegen in den Dateien `positives.csv` und `negatives.csv` im Ordner `data/raw`.

### 2. Geografische Daten: Eurostat GISCO

Für die Eingrenzung der Waldbrandbeobachtungen auf Italien verwenden wir die geografischen Landesgrenzen von Eurostat GISCO.

- Quelle: https://gisco-services.ec.europa.eu/distribution/v2/countries/
- Verwendete Datei: `IT-region-01m-4326-2024.geojson`
- Bezugsjahr der Landesgeometrie: 2024
- Koordinatensystem: EPSG:4326

### 3. Eigene Datenaufbereitung

Die ursprünglichen Mesogeos-Daten wurden für WildFireForecast mit Python verarbeitet.

Die Datenaufbereitung umfasst:

1. Einlesen und Untersuchung der Rohdatensätze.
2. Zeitliche Filterung auf 2015–2022.
3. Geografische Filterung auf Italien mithilfe von Eurostat GISCO.
4. Zusammenführung positiver und negativer Waldbrandbeobachtungen.
5. Prüfung der Datenqualität und potenzieller Data-Leakage-Variablen.
6. Zeitliche Aufteilung in Trainings-, Validierungs- und Testdaten.
7. Feature-Auswahl und Behandlung fehlender Werte.

Die ursprünglichen CSV-Dateien bleiben unverändert. Sämtliche Verarbeitungsschritte sind durch Python-Skripte im Ordner `src` dokumentiert und reproduzierbar.

### 4. Wissenschaftliche Quellenangabe

Kondylatos, S., Prapas, I., Camps-Valls, G., & Papoutsis, I. (2023). *Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean.* Advances in Neural Information Processing Systems, Datasets and Benchmarks Track.

Datensatz-Lizenz: Creative Commons Attribution 4.0 International (CC BY 4.0).

Die ursprünglichen Autorinnen und Autoren beziehungsweise Datenanbieter bleiben Urheber der zugrunde liegenden Daten. WildFireForecast ist eine eigenständige studentische Projektarbeit, die auf diesen Daten aufbaut.