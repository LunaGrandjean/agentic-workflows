"""Print the state, task errors, and output artifacts of a Vertex pipeline run."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.constants import GCP_PROJECT_ID, GCP_REGION


def main() -> None:
    from google.cloud import aiplatform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="Full projects/.../locations/.../pipelineJobs/... resource name.")
    args = parser.parse_args()
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    job = aiplatform.PipelineJob.get(args.job)
    print(f"State: {job.state.name}", flush=True)
    if job.gca_resource.error.message:
        print(f"Error: {job.gca_resource.error.message}", flush=True)
    for task in job.gca_resource.job_detail.task_details:
        print(f"{task.task_name}: {task.state.name}", flush=True)
        if task.error.message:
            print(f"  Error: {task.error.message}", flush=True)
        for name, output in task.outputs.items():
            for artifact in output.artifacts:
                print(f"  {name}: {artifact.uri}", flush=True)


if __name__ == "__main__":
    main()
