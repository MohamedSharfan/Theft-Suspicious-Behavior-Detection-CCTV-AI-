def generate_report(event):

    report = f"""
INCIDENT REPORT
-----------------------

Time: {event['timestamp']}
Camera: {event['camera']}
Person ID: {event['person_id']}
Threat Level: {event['status'].upper()}
Risk Score: {round(event['risk_score'] * 100, 1)}%

Reason:
{event['reason']}

AI Explanation:
{event['explanation']}

Alert Message:
{event['alert']}

Recommendation:
{event['recommendation']}
"""

    return report.strip()