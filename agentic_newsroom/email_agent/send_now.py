"""
Manual email trigger.

Run this whenever you want to send the latest report:
    python email_agent/send_now.py

Does not run ingestion or extraction — just sends whatever
report was generated in the last pipeline run.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from graph.knowledge_graph import KnowledgeGraph
from prediction.predictor import generate_prediction
from email_agent.sender import send_report

if __name__ == "__main__":
    print("Generating prediction signal...")
    kg         = KnowledgeGraph()
    prediction = generate_prediction(kg)

    print(f"Signal: {prediction['direction']} ({prediction['confidence']}) score={prediction['score']}")
    print(f"WTI: {prediction['signals']['eia']['wti_latest']}")
    print(f"Brent: {prediction['signals']['eia']['brent_latest']}")
    print()

    confirm = input("Send report to all recipients? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        sys.exit(0)

    result = send_report(prediction)
    print(f"\nDone: {result['sent']} sent, {result['failed']} failed")
    if result.get("errors"):
        for e in result["errors"]:
            print(f"  Failed: {e['email']} — {e['error']}")