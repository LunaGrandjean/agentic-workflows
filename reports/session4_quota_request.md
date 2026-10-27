# Session 4: previous T4 serving quota blocker

**Resolved for the practice on 27 October 2026.** After serving capacity became
available, model version 2 deployed successfully on one T4 in `europe-west2`.
Authenticated HTTP and Chainlit browser predictions passed, and the temporary
endpoint was deleted. See [Session 4 results](session4_results.md).

The request below is retained as a record of the earlier blocker. It does not need
to be submitted to finish this practice. Available quota can change as other teams
deploy or remove models; this run does not establish that the quota limit increased.

No quota-increase request was submitted by these scripts. The permission check found
that the current account could not update quotas or enable the Cloud Quotas API.
If a future deployment is blocked again, a teacher/project administrator can handle
the request below.

Open [Quotas and system limits for the course project](https://console.cloud.google.com/iam-admin/quotas?authuser=1&project=agentics-workflows-510809).
Select the Vertex AI / Agent Platform service, filter for T4 **serving** GPUs and
`europe-west2`, and request enough capacity for one additional available GPU. The
training quota used in Session 3 does not satisfy the online-serving quota.

Copy this request to the project administrator:

> Bonjour,
>
> Pour finaliser la pratique de la session 4, pourriez-vous rendre disponible un GPU
> NVIDIA T4 pour le **serving en ligne** dans le projet `agentics-workflows-510809`
> (numéro `123905721265`), région `europe-west2` ?
>
> Le déploiement échoue avec l'erreur `429 ResourceExhausted` sur le quota
> `CustomModelServingT4GPUsPerProjectPerRegion`.
>
> La configuration prévue est `n1-standard-8`, un seul T4, une seule réplique minimum
> et maximum. Le quota doit laisser au moins un T4 disponible pour ce test, en tenant
> compte de l'utilisation des autres équipes du projet.
>
> Les versions 1 et 2 du modèle `lulik-yoda-phi3` sont déjà enregistrées. L'endpoint
> temporaire sera supprimé après vérification pour arrêter les ressources de serving.
>
> Merci !

The read-only permission check returned none of `cloudquotas.quotas.update`,
`serviceusage.quotas.update`, `cloudquotas.quotas.get` or `serviceusage.services.enable`.
The Cloud Quotas API returned HTTP 403 `SERVICE_DISABLED` for this project. Evidence
is saved locally in `data/session4/serving_quota.json`.

Google documents the permissions for quota changes in [Quota permissions](https://docs.cloud.google.com/docs/quotas/permissions).
Requesting a quota adjustment does not itself deploy a model; the deployment must be
retried after the increased capacity is granted and available.

Once approved, run from `agentic-workflows/`:

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py deploy --model projects/123905721265/locations/europe-west2/models/8215968697160826880@2
.\.venv-session4\Scripts\python.exe scripts/test_online_endpoint.py
.\.venv-session4\Scripts\python.exe scripts/run_chat.py
```

Open `http://localhost:8000` and test a sentence. Stop the app with Ctrl+C, then remove
the serving endpoint:

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py cleanup
```
