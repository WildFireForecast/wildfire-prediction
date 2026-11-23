
"""
WildFireForecast

Zentrale Sicherheitssperre für die unabhängige
Testevaluation der Jahre 2021–2022.
"""

# ==================================================
# WISSENSCHAFTLICHE TESTFREIGABE
# ==================================================

# Solange die zeitliche Zieldefinition nicht
# abschliessend geklärt ist, bleibt der Test gesperrt.

FINAL_TEST_APPROVED = False


def require_final_test_approval():
    """
    Verhindert eine vorzeitige Testevaluation.

    Diese Funktion muss vor jedem Zugriff auf
    die unabhängigen Testdaten aufgerufen werden.
    """

    if not FINAL_TEST_APPROVED:

        raise RuntimeError(
            "\nFINALE TESTEVALUATION GESPERRT!\n"
            "Die zeitliche Zieldefinition ist noch "
            "nicht abschliessend bestätigt.\n"
            "Testdaten von 2021–2022 dürfen "
            "noch nicht für die Modellevaluation "
            "verwendet werden."
        )
