"""Synthetic multi-tenant page corpus.

Three fictional clients, each with a handful of pages — engineering runbooks,
HR policies, finance procedures. Small enough to encode in a minute on a warm
GPU cluster, varied enough to make multi-tenant filtering and visual retrieval
meaningful. Replace `PAGES` with your own pages (wiki export, Notion dump,
PDF batch, etc.) to point the demo at real content.
"""

import json
from pathlib import Path

PAGES = [
    # ── acme-corp: engineering ────────────────────────────────────────────
    {
        "client": "acme-corp",
        "page_id": "ACME-101",
        "title": "VPN setup for new engineers",
        "space": "Engineering",
        "author": "alice@acme",
        "web_url": "https://acme.atlassian.net/wiki/spaces/ENG/pages/101",
        "body": [
            "All engineers need to connect through the corporate VPN to reach internal services.",
            "We use Cisco AnyConnect on macOS and Windows, and the OpenConnect CLI on Linux.",
            "Download the client from it.acme.com/vpn, then sign in with your Okta credentials.",
            "Two-factor confirmation goes through Duo Push.",
            "If you hit a TLS error on first connection, check that the device certificate from Jamf is installed.",
            "For on-call rotations, request the always-on VPN profile from IT — it auto-reconnects after suspend.",
        ],
    },
    {
        "client": "acme-corp",
        "page_id": "ACME-102",
        "title": "On-call rotation and paging",
        "space": "Engineering",
        "author": "bob@acme",
        "web_url": "https://acme.atlassian.net/wiki/spaces/ENG/pages/102",
        "body": [
            "Engineering on-call runs Monday to Monday handovers at 10:00 PT.",
            "Primary takes the pager, secondary takes the laptop, both are paid the on-call stipend.",
            "Pages route through PagerDuty; the escalation policy is primary -> secondary (15 min) -> manager.",
            "During an incident open a Zoom bridge and a Slack channel named #inc-YYYYMMDD-summary.",
            "Postmortems are due within five working days and live in the Incidents space.",
        ],
    },
    {
        "client": "acme-corp",
        "page_id": "ACME-103",
        "title": "Deploying to production with our CI/CD pipeline",
        "space": "Engineering",
        "author": "carol@acme",
        "web_url": "https://acme.atlassian.net/wiki/spaces/ENG/pages/103",
        "body": [
            "We use GitHub Actions for CI and ArgoCD for delivery to Kubernetes.",
            "Merging to main triggers a build, runs the test suite, pushes an image to ECR, and updates the staging manifest.",
            "Production rollouts are gated by a manual approval in ArgoCD and require two reviewers from the service team.",
            "Use the rolling strategy with maxSurge=25% by default.",
            "Hotfix tags follow the pattern v1.2.3-hotfix.N and skip staging only with on-call approval recorded in the PR.",
        ],
    },
    {
        "client": "acme-corp",
        "page_id": "ACME-104",
        "title": "Local development setup",
        "space": "Engineering",
        "author": "dan@acme",
        "web_url": "https://acme.atlassian.net/wiki/spaces/ENG/pages/104",
        "body": [
            "Install mise to manage runtimes — it pins Node, Python, and Go versions per repo.",
            "Run `mise install` in the repo root, then `make dev` to spin up Postgres, Redis, and the API gateway in Docker.",
            "The seed data covers the last 30 days of staging traffic, sanitized of PII.",
            "If port 5432 is already taken, override DEV_PG_PORT in your shell profile.",
        ],
    },
    # ── globex: HR and admin ──────────────────────────────────────────────
    {
        "client": "globex",
        "page_id": "GLOBEX-201",
        "title": "Time off and vacation policy",
        "space": "HR",
        "author": "hr@globex",
        "web_url": "https://globex.atlassian.net/wiki/spaces/HR/pages/201",
        "body": [
            "Globex offers 25 working days of paid vacation per year, accruing monthly from the start date.",
            "Requests go through Workday at least two weeks in advance for absences longer than three days.",
            "Sick leave is separate and uncapped, but anything over three consecutive days requires a doctor's note.",
            "Parental leave is 18 weeks at full pay for the primary caregiver and 6 weeks for the secondary, regardless of gender.",
            "Unused vacation rolls over up to 10 days into the next calendar year; the rest is paid out.",
        ],
    },
    {
        "client": "globex",
        "page_id": "GLOBEX-202",
        "title": "Expense reports and reimbursement",
        "space": "HR",
        "author": "finance@globex",
        "web_url": "https://globex.atlassian.net/wiki/spaces/HR/pages/202",
        "body": [
            "Submit expenses in Expensify within 30 days of the transaction.",
            "Receipts are mandatory for any item over $25; below that, a description and category are enough.",
            "Travel bookings should go through Navan when possible — direct bookings need pre-approval from your manager.",
            "Reimbursements process every Friday and land in your payroll account the following Tuesday.",
            "Per diem for international travel is $80 USD equivalent for meals.",
        ],
    },
    {
        "client": "globex",
        "page_id": "GLOBEX-203",
        "title": "Office perks and meals",
        "space": "HR",
        "author": "office@globex",
        "web_url": "https://globex.atlassian.net/wiki/spaces/HR/pages/203",
        "body": [
            "Lunch is catered Monday through Thursday in the main cafe from 12:00 to 14:00.",
            "There are always vegetarian, vegan, and gluten-free options labeled at the buffet.",
            "Friday is a free-lunch credit you can spend at any partner restaurant in the office app.",
            "Snacks and drinks in the micro-kitchens are unlimited; please refill empty trays.",
            "The wellness stipend is $100 per month, claimable in Expensify under category Wellness.",
        ],
    },
    {
        "client": "globex",
        "page_id": "GLOBEX-204",
        "title": "Office Wi-Fi and guest network",
        "space": "IT",
        "author": "it@globex",
        "web_url": "https://globex.atlassian.net/wiki/spaces/IT/pages/204",
        "body": [
            "Connect to Globex-Corp for the employee network; sign in with your @globex.com SSO.",
            "Globex-Guest is for visitors — the rotating daily password is on the lobby screen.",
            "Printing requires the Globex-Print network and a one-time pairing with your laptop using the Mobility Print app.",
            "If your laptop will not join, forget the network and rejoin; the cert is renewed weekly and old caches get stuck.",
        ],
    },
    # ── initech: finance and compliance ───────────────────────────────────
    {
        "client": "initech",
        "page_id": "INIT-301",
        "title": "SOX controls and quarterly attestation",
        "space": "Compliance",
        "author": "compliance@initech",
        "web_url": "https://initech.atlassian.net/wiki/spaces/COMP/pages/301",
        "body": [
            "Initech is subject to SOX 404 reporting for financial controls over revenue, expense, and access management.",
            "Every quarter, control owners attest in AuditBoard that their controls operated as designed.",
            "Evidence is automatically collected from Workday, NetSuite, and Okta where possible; manual evidence goes in the AuditBoard Drive folder.",
            "External auditors test a sample of controls in Q3; expect requests for screenshots and approver lists.",
            "Exceptions must be logged within five business days of detection.",
        ],
    },
    {
        "client": "initech",
        "page_id": "INIT-302",
        "title": "Vendor onboarding and due diligence",
        "space": "Procurement",
        "author": "procurement@initech",
        "web_url": "https://initech.atlassian.net/wiki/spaces/PROC/pages/302",
        "body": [
            "New vendors above $50,000 annual spend require a security review and a SOC 2 Type II report on file.",
            "Submit the vendor questionnaire through Vanta; legal will review the MSA within five business days.",
            "Payment terms default to Net 60; faster terms require CFO approval and reduce the risk score in NetSuite.",
            "Sanctioned-country checks run automatically via the OFAC integration; any hit halts the workflow until cleared.",
            "Annual recertification of high-risk vendors happens every January.",
        ],
    },
    {
        "client": "initech",
        "page_id": "INIT-303",
        "title": "Audit prep checklist",
        "space": "Compliance",
        "author": "audit@initech",
        "web_url": "https://initech.atlassian.net/wiki/spaces/COMP/pages/303",
        "body": [
            "Two weeks before the auditors arrive, freeze the control population in AuditBoard and export the evidence index.",
            "Confirm with control owners that they will be available for walkthrough interviews — block 60 minutes in their calendars.",
            "Pull the user access review reports for the prior two quarters from Okta and confirm sign-off in writing.",
            "Have the change management JIRA queries ready: filter by label sox-relevant and status Done.",
            "If a control failed mid-period, document the compensating control and the date the gap was closed.",
        ],
    },
    {
        "client": "initech",
        "page_id": "INIT-304",
        "title": "Procurement card limits and exceptions",
        "space": "Procurement",
        "author": "procurement@initech",
        "web_url": "https://initech.atlassian.net/wiki/spaces/PROC/pages/304",
        "body": [
            "Procurement cards (P-cards) have a default monthly limit of $5,000 and a single-transaction limit of $1,500.",
            "Use them for low-dollar, low-risk purchases — software subscriptions and conference tickets are the common cases.",
            "Limit-increase requests need manager and CFO approval and a documented business need.",
            "Personal use, cash advances, and split transactions to bypass the single-transaction limit are policy violations.",
            "All P-card transactions reconcile in Coupa within 14 days of statement close.",
        ],
    },
]


def main():
    out = Path(__file__).resolve().parent / "pages.json"
    out.write_text(json.dumps(PAGES, indent=2))
    by_client = {}
    for p in PAGES:
        by_client[p["client"]] = by_client.get(p["client"], 0) + 1
    print(f"Wrote {len(PAGES)} pages to {out}")
    for client, n in sorted(by_client.items()):
        print(f"  {client}: {n} pages")


if __name__ == "__main__":
    main()
