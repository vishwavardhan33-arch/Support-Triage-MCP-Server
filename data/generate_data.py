"""
Generates a mock support-ticket dataset styled after a home-loan
partner-support inbox (loan status queries, document issues, disbursement
delays, KYC, portal bugs, escalations, etc).

Run: python generate_data.py
Writes: tickets.json (in this same folder)
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

CATEGORIES = {
    "Loan Status Query": [
        "Any update on loan application {ref}? It's been {days} days since submission.",
        "Customer is asking for a status update on file {ref}. Please advise on current stage.",
        "Need the sanction status for application {ref} before EOD, customer is following up.",
    ],
    "Document Upload Issue": [
        "Unable to upload the salary slip for {ref}, portal shows a generic error.",
        "The document upload keeps failing for Aadhaar/PAN on application {ref}.",
        "Getting 'file too large' error while uploading bank statements for {ref}.",
    ],
    "Disbursement Delay": [
        "Disbursement for {ref} was approved {days} days ago but funds haven't reached the customer.",
        "Partner is escalating disbursement delay on {ref}, customer is threatening to walk away.",
        "Sanctioned amount for {ref} not credited yet, please expedite.",
    ],
    "Interest Rate Query": [
        "Customer wants clarification on the interest rate slab applied to {ref}.",
        "Partner is asking why the rate for {ref} differs from the quoted rate on the app.",
    ],
    "Account Access": [
        "Partner unable to log in to the dashboard, OTP not being received.",
        "Getting 'account locked' message when trying to access the partner portal.",
    ],
    "Partner Onboarding": [
        "New DSA partner onboarding stuck at KYC verification step for {days} days.",
        "Partner agreement signed but portal access not yet provisioned.",
    ],
    "KYC Issue": [
        "KYC mismatch flagged for {ref} — name on PAN doesn't match application.",
        "Customer's KYC re-verification pending, blocking progress on {ref}.",
    ],
    "Payment Reconciliation": [
        "Payout for {ref} doesn't match the commission sheet, please reconcile.",
        "Partner says payout for last month is short by a fixed processing fee, needs review.",
    ],
    "Technical / Portal Bug": [
        "Loan calculator on the partner portal is showing incorrect EMI for {ref}.",
        "Dashboard is not loading application {ref}, spinning indefinitely.",
    ],
    "Complaint - Escalation": [
        "Customer is extremely unhappy with the delay on {ref} and wants a callback today.",
        "Partner is threatening to stop referring leads due to repeated delays on cases like {ref}.",
    ],
}

SENDER_TYPES = ["partner", "customer"]
STATUSES = ["open", "in_progress", "closed"]
PRIORITY_BY_CATEGORY = {
    "Disbursement Delay": "high",
    "Complaint - Escalation": "high",
    "KYC Issue": "medium",
    "Payment Reconciliation": "medium",
    "Loan Status Query": "low",
    "Document Upload Issue": "medium",
    "Interest Rate Query": "low",
    "Account Access": "medium",
    "Partner Onboarding": "medium",
    "Technical / Portal Bug": "low",
}

FIRST_NAMES = ["Rohan", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Karan",
               "Neha", "Arjun", "Divya", "Suresh", "Meera"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Joshi", "Rao"]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def build_ticket(i, base_date):
    category = random.choice(list(CATEGORIES.keys()))
    template = random.choice(CATEGORIES[category])
    ref = f"BHL-{random.randint(10000, 99999)}"
    days = random.randint(1, 12)
    body = template.format(ref=ref, days=days)
    sender_type = random.choice(SENDER_TYPES)
    status = random.choices(STATUSES, weights=[0.4, 0.35, 0.25])[0]
    timestamp = base_date - timedelta(hours=random.randint(0, 24 * 20))

    return {
        "id": f"T-{1000 + i}",
        "subject": f"{category}: {ref}",
        "body": body,
        "sender": random_name(),
        "sender_type": sender_type,
        "reference_id": ref,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "category": category,          # ground-truth label (for evaluating the classifier later)
        "priority": PRIORITY_BY_CATEGORY[category],
        "status": status,
    }


def main():
    base_date = datetime(2026, 9, 16, 10, 0, 0)
    tickets = [build_ticket(i, base_date) for i in range(40)]
    tickets.sort(key=lambda t: t["timestamp"], reverse=True)

    out_path = Path(__file__).parent / "tickets.json"
    with open(out_path, "w") as f:
        json.dump(tickets, f, indent=2)

    print(f"Wrote {len(tickets)} tickets to {out_path}")


if __name__ == "__main__":
    main()
