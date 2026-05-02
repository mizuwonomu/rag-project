import os
import csv
from datetime import datetime

FEEDBACK_CSV = "feedback_log.csv"

def save_feedback(question, answer, rating, reason="", comment=""):
    file_exists = os.path.isfile(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Question", "Answer", "Rating", "Reason", "Comment"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer, rating, reason, comment])