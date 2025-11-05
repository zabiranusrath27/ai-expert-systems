# src/rules.py
def rule_based_recommendation(row: dict) -> str:
    temp = row.get("Air temperature [K]", 300)
    proc = row.get("Process temperature [K]", 300)
    torque = row.get("Torque [Nm]", 30)
    wear = row.get("Tool wear [min]", 0)

    if wear > 200:
        return "High tool wear → schedule tool replacement (TPM)."
    if temp > 320 and proc > 320:
        return "Abnormal thermal load → check lubrication/cooling (ISO 9001:2015)."
    if torque > 60:
        return "Excessive torque → inspect alignment/bearing."
    return "Machine within expected operating range."
