
"""
WildFireForecast

Zentrale Freigabe für die finale Evaluation
auf den unabhängigen Testdaten 2021–2022.

Die Ergebnisse werden als binäre Klassifikation
der Mesogeos-Waldbrand-Samples interpretiert.

Die zeitlichen Einschränkungen der Zieldefinition
werden im wissenschaftlichen Bericht dokumentiert.
"""

# Nur nach ausdrücklicher Teamfreigabe aktivieren.
FINAL_TEST_APPROVED = False


def require_final_test_approval():
    """
    Verhindert Testdatenzugriff ohne Freigabe.
    """

    if not FINAL_TEST_APPROVED:
        raise RuntimeError(
            "FINALE TESTEVALUATION GESPERRT.\n"
            "Eine ausdrückliche Freigabe ist erforderlich."
        )
