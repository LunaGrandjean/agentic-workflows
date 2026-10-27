"""Read the exact regional T4 serving quota that can block Session 4 deployment."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from agentic_workflows.constants import GCP_PROJECT_ID, GCP_PROJECT_NUMBER, GCP_REGION

    project = GCP_PROJECT_NUMBER or GCP_PROJECT_ID
    quota_id = "CustomModelServingT4GPUsPerProjectPerRegion"
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=GCP_PROJECT_ID,
    )
    url = f"https://cloudquotas.googleapis.com/v1/projects/{project}/locations/global/services/aiplatform.googleapis.com/quotaInfos/{quota_id}"
    session = AuthorizedSession(credentials)
    permissions = session.post(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{GCP_PROJECT_ID}:testIamPermissions",
        json={"permissions": ["serviceusage.services.enable", "cloudquotas.quotas.get", "cloudquotas.quotas.update", "serviceusage.quotas.update"]},
        timeout=30,
    )
    response = session.get(url, timeout=30)
    result = {
        "project": project, "region": GCP_REGION, "quota_id": quota_id,
        "http_status": response.status_code, "response": response.json(),
        "quota_permissions_http_status": permissions.status_code, "quota_permissions": permissions.json(),
    }
    output = PROJECT_ROOT / "data/session4/serving_quota.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
