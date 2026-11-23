
from pathlib import Path
import subprocess
import sys

from test_guard import FINAL_TEST_APPROVED


# ==================================================
# 1. PROJEKTPFADE
# ==================================================

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    "Logistische Regression": "test_logistic.py",
    "XGBoost": "test_xgboost.py",
    "LSTM": "test_lstm.py",
    "Transformer": "test_transformer.py",
}


# ==================================================
# 2. SKRIPTE KONTROLLIEREN
# ==================================================

def check_files():
    print("\nPrüfe Testskripte ...")

    for name, filename in SCRIPTS.items():
        path = TEST_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Fehlendes Skript: {path}"
            )

        print(f"{name}: vorhanden")

    print("\nAlle vier Skripte vorhanden.")


# ==================================================
# 3. MODELLPRÜFUNG OHNE TESTDATEN
# ==================================================

def check_all_models():
    print("\nStarte die Prüfung aller acht Modelle.")

    for name, filename in SCRIPTS.items():
        print("\n" + "-" * 60)
        print(f"Prüfe: {name}")
        print("-" * 60, flush=True)

        subprocess.run(
            [
                sys.executable,
                str(TEST_DIR / filename),
            ],
            cwd=ROOT,
            check=True,
        )

    print("\n" + "=" * 60)
    print("ALLE ACHT MODELLE ERFOLGREICH GEPRÜFT")
    print("=" * 60)
    print("Keine unabhängigen Testdaten verwendet.")


# ==================================================
# 4. FINALE TESTEVALUATION
# ==================================================

def execute_final_test():
    # Freigabe VOR dem Start jedes Modellprozesses.
    if not FINAL_TEST_APPROVED:
        raise RuntimeError(
            "TEST GESPERRT: Die zeitliche "
            "Zieldefinition ist noch nicht bestätigt."
        )

    print("\nStarte finale Testevaluation ...")

    for name, filename in SCRIPTS.items():
        print(f"\nEvaluiere: {name}", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(TEST_DIR / filename),
                "--execute",
            ],
            cwd=ROOT,
            check=True,
        )

    print("\nAlle acht Modelle wurden evaluiert.")
    print("Ergebnisse müssen noch zusammengeführt werden.")


# ==================================================
# 5. HAUPTPROGRAMM
# ==================================================

def main():
    print("=" * 60)
    print("WILDFIREFORECAST - TESTSTEUERUNG")
    print("=" * 60)

    check_files()

    if "--check" in sys.argv:
        check_all_models()
        return

    if "--execute" in sys.argv:
        execute_final_test()
        return

    print("\nKeine Evaluation gestartet.")

    if not FINAL_TEST_APPROVED:
        print("STATUS: FINALE TESTEVALUATION GESPERRT")
        print("Die zeitliche Zieldefinition ist noch offen.")

    print("\nVerfügbare Befehle:")
    print("  --check    Modelle ohne Testdaten prüfen")
    print("  --execute  Finalen Test nach Freigabe starten")


if __name__ == "__main__":
    main()
