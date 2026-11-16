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

## 3. Machine-Learning-Modelle

Im Projekt WildFireForecast werden drei Modellfamilien mit insgesamt sechs
Modellvarianten untersucht. Die Entwicklung und vorläufige Bewertung
erfolgen auf einer gemeinsamen Datengrundlage für Italien.

### 3.1 Logistische Regression

- **Modell 1A:** Ohne die Oberflächentemperaturvariablen `lst_day` und `lst_night`.
- **Modell 1B:** Mit beiden Oberflächentemperaturvariablen.

Die Modelle verwenden eine scikit-learn-Pipeline mit Imputation
fehlender Werte und Standardisierung der Eingabemerkmale.

Die bisherigen Validierungsergebnisse aus dem Jahr 2020 sind
vorläufig und werden im wissenschaftlichen Projektbericht dokumentiert.

#### Validierungsergebnisse 2020

| Kennzahl | Modell 1A | Modell 1B |
|---|---:|---:|
| Accuracy | 84,23 % | 84,79 % |
| Precision | 83,59 % | 84,46 % |
| Recall | 87,17 % | 87,17 % |
| F1-Score | 85,34 % | 85,79 % |
| ROC-AUC | 90,60 % | 90,91 % |

Modell 1B erreicht bei gleichem Recall eine leicht bessere Precision und verursacht zwei Fehlalarme weniger. Es ist deshalb unsere vorläufig bevorzugte Variante.

Diese Ergebnisse stammen ausschliesslich aus der Validierung. Die abschliessende Testauswertung wurde noch nicht durchgeführt.

### 3.2 XGBoost

- **Modell 2A:** XGBoost ohne zusätzliches Feature Engineering.
- **Modell 2B:** XGBoost mit zusätzlichem Feature Engineering.

Für beide Modellvarianten wurden Baseline-Modelle trainiert,
validiert und anschliessend Hyperparameter-Tuning durchgeführt.

Die gespeicherten Baseline- und Tuning-Modelle befinden sich
in getrennten Unterordnern.

### 3.3 Long Short-Term Memory (LSTM)

- **Modell 3A:** LSTM ohne die Merkmale `lst_day` und `lst_night`.
- **Modell 3B:** LSTM mit diesen beiden Merkmalen.

Die LSTM-Modelle verarbeiten Eingabesequenzen mit 59 Zeitschritten.
Die trainierten Modellgewichte und zugehörigen Preprocessing-Objekte
werden gemeinsam gespeichert.

Auch für die LSTM-Modelle wurde Hyperparameter-Tuning durchgeführt.
Die zugehörigen Ergebnisse werden unter `reports/lstm/tuning/` abgelegt.

### 3.4 Methodische Einschränkungen

Das bisherige Eingabefenster umfasst `time_idx=0–58` (59 Tage).
Die Annahme, dass `time_idx=59` den vorherzusagenden Zieltag
repräsentiert, ist noch nicht abschliessend wissenschaftlich verifiziert.

Die Testdaten der Jahre 2021–2022 wurden für die bisherigen
Modellvergleiche nicht verwendet. Die Ergebnisse beziehen sich
auf die Validierungsdaten des Jahres 2020.

## 4. Projektstruktur

```text
WildFireForecast/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── reference/
├── src/
│   ├── preprocessing/
│   ├── validation_checks/
│   └── models/
│       ├── logistic_regression/
│       ├── xgboost/
│       │   └── tuning/
│       └── lstm/
│           └── tuning/
├── models/
│   ├── logistic_regression/
│   │   └── baseline/
│   ├── xgboost/
│   │   ├── baseline/
│   │   └── tuned/
│   └── lstm/
│       └── baseline/
├── reports/
│   ├── logistic_regression/
│   │   └── baseline/
│   ├── xgboost/
│   │   ├── baseline_2A/
│   │   ├── baseline_2B/
│   │   └── tuning/
│   └── lstm/
│       ├── baseline/
│       └── tuning/
├── docs/
│   └── WildFireForecast_Projektbericht.docx
├── README.md
└── requirements.txt
```

Die Verzeichnisse sind nach ihrer Funktion gegliedert:

- `data/`: Originaldaten, Zwischenstände und vorbereitete Datensätze.
- `src/preprocessing/`: Datenfilterung und Aufbereitung.
- `src/validation_checks/`: Prüfungen der Daten und Zielvariablen.
- `src/models/`: Training, Analyse und Hyperparameter-Tuning.
- `models/`: Gespeicherte Modelle und Preprocessing-Objekte.
- `reports/`: Validierungskennzahlen, Grafiken und Tuning-Ergebnisse.
- `docs/`: Wissenschaftlicher Projektbericht.

## 5. Projektstatus

Folgende Arbeiten wurden bereits durchgeführt:

- Geografische Eingrenzung auf Italien.
- Datenbereinigung und zeitliche Aufteilung.
- Entwicklung und Validierung der logistischen Regressionsmodelle 1A und 1B.
- Entwicklung und Validierung der XGBoost-Modelle 2A und 2B.
- Entwicklung und Validierung der LSTM-Modelle 3A und 3B.
- Hyperparameter-Tuning für XGBoost und LSTM.
- Umstrukturierung der Modell- und Ergebnisordner.
- Prüfung der gespeicherten Modellartefakte auf Ladefähigkeit.

Noch ausstehend sind insbesondere die abschliessende Prüfung
der zeitlichen Zielzuordnung, die finale Modellevaluation
auf dem zurückgehaltenen Testdatensatz und die wissenschaftliche
Gesamtinterpretation der Ergebnisse.

## 6. Projektdokumentation

Der ausführliche Projektbericht befindet sich im Ordner `docs/`. Dort werden die Datenquellen, Entscheidungen, methodischen Überlegungen und bisherigen Resultate dokumentiert.

Der Bericht wird während der weiteren Projektentwicklung ergänzt.