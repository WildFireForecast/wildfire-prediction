# WildFireForecast

## Waldbrandvorhersage in Italien mit Machine Learning

### Projektbeschreibung

Im Projekt WildFireForecast untersuchen wir, ob sich Waldbrandereignisse anhand historischer Wetter-, Vegetations-, Gelände- und Umweltdaten mit Machine Learning vorhersagen lassen.

Dafür entwickeln und vergleichen wir fünf unterschiedliche Machine-Learning-Modelle. Unser Ziel ist es, herauszufinden, wie gut die Modelle zwischen positiven und negativen Waldbrand-Samples unterscheiden können und welche Einflussgrössen für die Vorhersagen relevant sind.

Wir haben uns bewusst auf **Italien** beschränkt, um das Projekt überschaubar zu halten und die verschiedenen Modelle auf einer gemeinsamen Datengrundlage vergleichen zu können.

## 1. Datenquellen

### Waldbrand- und Umweltdaten

Die Originaldaten stammen aus dem Forschungsdatensatz *A dataset for time-series wildfire danger forecasting* von Spyros Kondylatos, veröffentlicht 2025 auf Zenodo.

Quelle: https://doi.org/10.5281/zenodo.17201754

Der Datensatz baut auf der Mesogeos-Forschung auf und enthält unter anderem Wetterdaten, Vegetationsinformationen, Gelände- und Landnutzungsmerkmale sowie Informationen zu Waldbränden.

Wissenschaftlicher Hintergrund:

Kondylatos et al. (2023): *Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean.*

https://arxiv.org/abs/2306.05144

Für die geografische Eingrenzung auf Italien verwenden wir zudem Landesgrenzen von Eurostat GISCO.

## 2. Datenaufbereitung

Aus den ursprünglichen Dateien `positives.csv` und `negatives.csv` wurden ausschliesslich Samples für Italien herausgefiltert.

Dabei ergaben sich:

| Kategorie | Anzahl Samples |
|---|---:|
| Positive Samples | 2'280 |
| Negative Samples | 1'952 |
| Insgesamt vor zeitlicher Bereinigung | 4'232 |
| Nach zeitlicher Bereinigung | 4'206 |

Die Original-Zeitreihen enthalten jeweils 60 Tage. In unserer bisherigen Datenaufbereitung werden die Zeitindizes 0 bis 58 als Eingaben verwendet. Index 59 wird vorläufig als Zieltag angenommen.

Die genaue zeitliche Zuordnung zum Waldbrandereignis muss noch abschliessend bestätigt werden.

### Zeitliche Aufteilung

| Datengruppe | Zeitraum |
|---|---|
| Training | 2006–2019 |
| Validierung | 2020 |
| Test | 2021–2022 |

Samples, deren Zeitreihen eine Grenze zwischen diesen Zeiträumen überschreiten, wurden ausgeschlossen.

## 3. Bisherige Modelle

### Modell 1 – Logistische Regression

Wir haben zwei Varianten der logistischen Regression entwickelt:

- **Modell 1A:** Ohne die Oberflächentemperaturvariablen `lst_day` und `lst_night`.
- **Modell 1B:** Mit diesen beiden Variablen.

Fehlende Eingabewerte werden mit dem Median der Trainingsdaten ersetzt. Anschliessend werden die Merkmale standardisiert.

Die Vorverarbeitung und das Modell sind in einer gemeinsamen scikit-learn-Pipeline implementiert.

### Validierungsergebnisse 2020

| Kennzahl | Modell 1A | Modell 1B |
|---|---:|---:|
| Accuracy | 84,23 % | 84,79 % |
| Precision | 83,59 % | 84,46 % |
| Recall | 87,17 % | 87,17 % |
| F1-Score | 85,34 % | 85,79 % |
| ROC-AUC | 90,60 % | 90,91 % |

Modell 1B erreicht bei gleichem Recall eine leicht bessere Precision und verursacht zwei Fehlalarme weniger. Es ist deshalb unsere vorläufig bevorzugte Variante.

Diese Ergebnisse stammen ausschliesslich aus der Validierung. Die abschliessende Testauswertung wurde noch nicht durchgeführt.

## 4. Aktuelle Python-Skripte

Die aktuelle Verarbeitung befindet sich im Ordner `src/`.

| Datei | Funktion |
|---|---|
| `01_filter_italy.py` | Geografische Eingrenzung auf Italien |
| `02_check_data.py` | Datenprüfung |
| `03_split_samples.py` | Zeitliche Aufteilung |
| `04_prepare_inputs.py` | Aufbereitung der Eingabezeitreihen |
| `05_build_baseline_tables.py` | Erstellung der Modelltabellen |
| `06_train_logistic.py` | Training und Validierung von Modell 1 |
| `07_check_target_day.py` | Prüfung der Brandinformationen über die Zeit |
| `08_check_labels.py` | Überprüfung der Zielvariablen |
| `09_check_label_over_time.py` | Prüfung der zeitlichen Konstanz der Kennzeichnung |

## 5. Projektstatus

**Abgeschlossen:** Datenfilterung für Italien, zeitliche Datenaufteilung, erste Datenqualitätsprüfungen, Training und Validierung der logistischen Regression.

**Noch offen:** Abschliessende Prüfung des Zielzeitpunkts, weitere vier ML-Modelle, Modellvergleich, Testevaluation und Schlussfolgerungen.

Die derzeitigen Modellergebnisse sind als vorläufig gekennzeichnet.

## 6. Projektdokumentation

Der ausführliche Projektbericht befindet sich im Ordner `docs/`. Dort werden die Datenquellen, Entscheidungen, methodischen Überlegungen und bisherigen Resultate dokumentiert.

Der Bericht wird während der weiteren Projektentwicklung ergänzt.