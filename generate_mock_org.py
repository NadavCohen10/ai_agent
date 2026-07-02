"""
Generates data/mock_organization/ with realistic corporate security policy files.
Run from the project root: python generate_mock_org.py
"""

import os

OUTPUT_DIR = os.path.join("data", "mock_organization")

FILES = {
    "Access_Control_Policy.txt": """\
ACCESS CONTROL POLICY
Organization: Acme Corp | Document ID: POL-AC-001 | Version: 3.2 | Effective: 2024-01-15

1. PASSWORD REQUIREMENTS
All user accounts must be protected by passwords of a minimum length of 14 characters, containing
at least one uppercase letter, one lowercase letter, one numeric digit, and one special character
(!@#$%^&*). Passwords must be rotated every 90 days and cannot be reused within the last 12
generations. The use of the user's name, email address, or the word "password" in any form is
strictly prohibited. Failed login attempts are locked after 5 consecutive failures, triggering a
30-minute account lockout and an automated alert to the Security Operations Center (SOC).

2. MULTI-FACTOR AUTHENTICATION (MFA)
MFA is mandatory for all employee accounts without exception, including service accounts with
interactive login capabilities. The approved second factors are: TOTP authenticator apps (Google
Authenticator, Microsoft Authenticator), hardware security keys (FIDO2/WebAuthn), or SMS OTP as
a last-resort fallback for non-privileged users only. Privileged Access Workstations (PAWs) used
by system administrators must use hardware security keys exclusively. MFA bypass is forbidden and
any request to disable MFA must be approved in writing by the CISO and documented in the change
management system.

3. ROLE-BASED ACCESS CONTROL (RBAC) & LEAST PRIVILEGE
Access to systems and data is granted on a least-privilege basis using a formal Role-Based Access
Control (RBAC) model. All access requests must be submitted via the IT Service Desk ticketing
system and approved by both the employee's direct manager and the relevant system owner. Access
rights are reviewed quarterly by department heads and the IT Security team. Upon employee
termination, all access must be revoked within 4 business hours of the HR notification. Privileged
accounts (domain admin, root, DBA) are subject to Privileged Access Management (PAM) controls and
all sessions are recorded and retained for 12 months.
""",

    "Incident_Response_Plan.txt": """\
INCIDENT RESPONSE PLAN
Organization: Acme Corp | Document ID: POL-IR-002 | Version: 2.1 | Effective: 2023-11-01

1. INCIDENT CLASSIFICATION & SEVERITY LEVELS
All security incidents are classified into four severity tiers upon detection. Severity 1 (Critical)
includes active ransomware, confirmed data breaches affecting more than 500 records, or complete
outage of a critical production system; response must begin within 15 minutes of detection.
Severity 2 (High) covers unauthorized access to sensitive systems or malware containment events;
response SLA is 1 hour. Severity 3 (Medium) includes policy violations or failed intrusion attempts;
response SLA is 4 hours. Severity 4 (Low) covers informational alerts requiring investigation;
response SLA is 24 hours. All incidents, regardless of severity, must be logged in the SIEM and
assigned a ticket in the incident tracking system within 30 minutes of classification.

2. CONTAINMENT, ERADICATION & RECOVERY
Upon confirmation of a Severity 1 or 2 incident, the Incident Response Team (IRT) must isolate
affected systems from the network within 30 minutes using automated EDR quarantine or manual VLAN
segregation. No affected system may be wiped or reimaged before a forensic image is captured and
preserved to a write-protected evidence store. Eradication steps must be documented in real time in
the incident ticket. Recovery to production is gated on a sign-off from both the IRT Lead and the
system owner. Full root-cause analysis (RCA) must be completed and submitted to the CISO within
5 business days of incident closure.

3. NOTIFICATION & REPORTING OBLIGATIONS
Any incident involving personal data of EU residents must be assessed for GDPR breach notification
requirements within 24 hours of discovery. If a reportable breach is confirmed, the relevant
supervisory authority must be notified within 72 hours per Article 33 of the GDPR. The Legal and
Compliance team must be engaged immediately for any incident classified Severity 1 or 2. External
communication, including press statements or customer notifications, may only be issued by the
VP of Communications in coordination with Legal. A post-incident report must be distributed to
executive leadership within 10 business days of closure.
""",

    "Network_Architecture_Overview.txt": """\
NETWORK ARCHITECTURE & SECURITY OVERVIEW
Organization: Acme Corp | Document ID: DOC-NET-005 | Version: 1.8 | Effective: 2024-03-01

1. NETWORK SEGMENTATION
The Acme Corp network is divided into five security zones enforced by next-generation firewalls
(Palo Alto PA-5250 cluster in active-active HA): External DMZ, Internal DMZ, Corporate LAN,
Production Environment, and Management Network. All inter-zone traffic is denied by default and
explicitly permitted only by approved firewall rules reviewed quarterly. The Production Environment
is further segmented by application tier (Web, Application, Database) using micro-segmentation
enforced by NSX-T. No direct communication is permitted between the Corporate LAN and the
Production Database tier; all access is brokered through the Application tier or a designated
jump server in the Management Network.

2. PERIMETER & ENDPOINT SECURITY
All internet-bound traffic is routed through a redundant pair of Zscaler Internet Access (ZIA)
nodes acting as a cloud-based Secure Web Gateway (SWG), providing TLS inspection, URL filtering,
and inline CASB capabilities. Remote access for employees is provided exclusively via Zscaler
Private Access (ZPA) using a Zero Trust Network Access (ZTNA) model; traditional VPN is deprecated
and disabled. All endpoints must run CrowdStrike Falcon agent (Prevent + Insight modules) and
must pass a device posture check (OS patch level within 30 days, disk encryption enabled, agent
running) before ZPA access is granted. Unmanaged devices are categorically denied access to
internal resources.

3. ENCRYPTION & CERTIFICATE MANAGEMENT
All data in transit across any network boundary must be encrypted using TLS 1.2 or higher; TLS 1.0
and 1.1 are explicitly disabled on all load balancers and application servers. Internal services
communicate over mTLS where supported. Public-facing TLS certificates are managed via DigiCert
with a maximum validity of 398 days and are renewed automatically 30 days before expiration using
ACME protocol integration. All data at rest in the Production Environment is encrypted using
AES-256; encryption key management is handled by HashiCorp Vault with AWS KMS as the seal backend.
Encryption key rotation is performed annually for data-at-rest keys and every 90 days for
transit/session keys.
""",

    "Data_Backup_Procedure.txt": """\
DATA BACKUP & RECOVERY PROCEDURE
Organization: Acme Corp | Document ID: PROC-BK-003 | Version: 2.4 | Effective: 2023-09-01

1. BACKUP SCHEDULE & RETENTION
All production databases and file servers are subject to the following backup schedule: full backups
are performed every Sunday at 01:00 UTC, differential backups every weekday night at 02:00 UTC,
and transaction log backups every 4 hours for all SQL-based databases. Backup retention periods
are as follows: daily backups are retained for 30 days, weekly backups for 90 days, and monthly
backups (taken on the first Sunday of each month) for 12 months. Backups of systems that process
payment card data (PCI-DSS scope) are retained for a minimum of 12 months in accordance with
PCI-DSS Requirement 9.8.1. All retention schedules are enforced by automated lifecycle policies
in the backup platform and reviewed annually by the IT Manager.

2. BACKUP STORAGE & ENCRYPTION
All backups are written to a primary Veeam Backup & Replication repository located in the on-premises
data center and replicated within 2 hours to an immutable S3-compatible object storage bucket in
AWS (us-east-1 region). The offsite S3 bucket is configured with Object Lock in Compliance mode
with a 30-day retention lock, ensuring backups cannot be deleted or overwritten even by
administrators. All backup data is encrypted at rest using AES-256 and in transit using TLS 1.2.
Backup encryption keys are stored separately from the backup data in HashiCorp Vault and are
rotated annually. Access to the backup repository console is restricted to the Backup Administrator
role and requires MFA.

3. RECOVERY TESTING & RTO/RPO TARGETS
Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) are defined per system tier.
Tier 1 (mission-critical: ERP, payment processing) has an RTO of 4 hours and an RPO of 1 hour.
Tier 2 (business-critical: CRM, email) has an RTO of 8 hours and an RPO of 4 hours. Tier 3
(standard: internal wikis, dev environments) has an RTO of 24 hours and an RPO of 24 hours.
Full recovery tests are conducted quarterly for all Tier 1 systems and semi-annually for Tier 2
systems, with results documented and reviewed by the IT Manager and CISO. Any failed recovery
test automatically triggers a remediation plan with a 30-day resolution deadline.
""",

    "Vulnerability_Management_Policy.txt": """\
VULNERABILITY MANAGEMENT POLICY
Organization: Acme Corp | Document ID: POL-VM-004 | Version: 1.5 | Effective: 2024-02-01

1. SCANNING FREQUENCY & COVERAGE
All internet-facing systems and internal production assets must be scanned for vulnerabilities on
a continuous basis using Tenable.io. Authenticated network scans cover the entire RFC-1918 address
space and are executed weekly; unauthenticated scans of the External DMZ are executed daily.
Container images in the CI/CD pipeline are scanned at build time using Trivy and any image with
a Critical or High severity finding is blocked from being pushed to the production registry.
All new assets must be registered in the CMDB and added to the scan scope within 5 business days
of provisioning. Scan results are automatically ingested into the vulnerability management platform
and correlated with the CMDB for asset ownership assignment.

2. REMEDIATION SLAs
Vulnerabilities are prioritized using the CVSS v3.1 base score in combination with CISA KEV catalog
status and asset criticality. Mandatory remediation SLAs are as follows: Critical (CVSS 9.0-10.0)
must be remediated within 7 calendar days; High (CVSS 7.0-8.9) within 30 calendar days; Medium
(CVSS 4.0-6.9) within 90 calendar days; Low (CVSS 0.1-3.9) within 180 calendar days. Any Critical
vulnerability listed in the CISA KEV catalog has an accelerated SLA of 48 hours regardless of
asset tier. Exceptions to SLAs require written approval from the CISO, must include a documented
compensating control, and are valid for a maximum of 30 days before re-approval is required.

3. PENETRATION TESTING
An external penetration test of all internet-facing applications and infrastructure must be conducted
at minimum annually by a CREST-accredited third-party provider. An internal penetration test
(assumed-breach scenario) must be performed semi-annually. All critical and high findings from
penetration tests must be remediated within the same SLAs defined in section 2. A re-test must be
performed by the same provider within 60 days of remediation to validate closure. Penetration test
reports are classified as Confidential and distributed only to the CISO, CTO, and relevant system
owners. Bug bounty submissions via the HackerOne program are triaged within 5 business days and
follow the same remediation SLAs.
""",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, content in FILES.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  created: {path}")
    print(f"\nDone — {len(FILES)} files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
