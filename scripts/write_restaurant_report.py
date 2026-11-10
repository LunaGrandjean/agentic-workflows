"""Write a factual project report from local audit files and actual downloaded runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/restaurant")
    args = parser.parse_args()
    data = args.input_dir
    manifest = json.loads((data / "raw/manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((data / "prepared/preparation.json").read_text(encoding="utf-8"))
    lines = ["# Projet final — Restaurant Studio", "",
        f"État documenté le {datetime.now(ZoneInfo('Europe/Paris')).strftime('%d/%m/%Y')}. Les résultats ci-dessous proviennent des fichiers réellement produits.", "",
        "## Données vérifiées", "",
        "Dataset : [GEM/e2e_nlg](https://huggingface.co/datasets/GEM/e2e_nlg), anglais, licence CC-BY-SA-4.0.",
        f"Révision source : `{manifest['source_revision']}`. Révision Parquet : `{manifest['parquet_revision']}`.", "",
        "| Split officiel | Lignes source | MR uniques | Lignes préparées, corpus complet |",
        "| --- | ---: | ---: | ---: |"]
    for split, counts in audit["splits"].items():
        lines.append(f"| {split} | {counts['raw_rows']} | {counts['unique_mrs']} | {counts['prepared_rows']} |")
    quality_path = ROOT / "reports/restaurant_data_quality.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if quality["snapshot"] != manifest["parquet_revision"]:
            raise ValueError("Refresh the data-quality audit for the selected snapshot.")
        lines += ["", f"MR avec valeurs contradictoires pour un même slot : `{json.dumps(quality['conflicting_slot_rows'])}`.",
            f"Références textuelles identiques entre splits, après normalisation casse/espaces : `{json.dumps(quality['reference_overlap'])}`.",
            "Ces diagnostics sont distincts des chevauchements d'entrées ci-dessous. Le pilote conserve le corpus officiel ; aucun filtrage des contradictions n'a été expérimenté."]
    lines += ["", f"Chevauchements après normalisation : `{json.dumps(audit['canonical_overlap_before_filtering'])}`.",
        f"Lignes train retirées pour chevauchement : {audit['training_rows_removed_for_overlap']}.",
        "Les répétitions de slots dans le corpus sont conservées et signalées ; elles ne sont pas silencieusement écrasées.", "",
        "## Ce qui est adapté", "",
        "Préparation reproductible, splits officiels, prompt Restaurant commun au train et au serving, validation dédiée, comparaison base/LoRA, ROUGE multi-références, registre et endpoints séparés de Yoda, application Chainlit et exemples de démonstration.", "",
        "Le pipeline de test dédié évalue des adaptateurs figés sans réentraîner. Le code des sessions Yoda reste disponible.", "",
        "## Exécutions et résultats", ""]
    runs = sorted((data / "runs").glob("*.json"))
    if runs:
        for path in runs:
            run = json.loads(path.read_text(encoding="utf-8"))
            lines += [f"- Run soumis : `{run['job']}` ; smoke test : `{run['smoke_test']}` ; dernier état enregistré : `{run['state']}`."]
    else:
        lines.append("Aucun run Vertex soumis n'est enregistré localement.")
    summary_path = data / "run_summary.json"
    results_path = data / "evaluation_results.csv"
    if summary_path.exists() and results_path.exists():
        import pandas as pd
        from agentic_workflows.restaurant_evaluation import score_restaurant_predictions

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(results_path, keep_default_na=False)
        _, metrics = score_restaurant_predictions(frame)
        training = summary.get("artifacts", {}).get("fine-tuning-component/metrics", {}).get("metadata", {})
        parameters = summary.get("parameters", {})
        lines += ["", f"Run téléchargé : `{summary['job']}` ; état : `{summary['state']}`.",
            f"Taille train demandée : {parameters.get('train_limit', 'voir métadonnées')} ; validation : {parameters.get('validation_limit', 'voir métadonnées')} ; époques configurées : {parameters.get('num_train_epochs', 'voir métadonnées')} ; learning rate : {parameters.get('learning_rate', 'voir métadonnées')}.",
            f"Évaluation sur {len(frame)} MR ; split(s) : {', '.join(sorted(frame['source_split'].unique()))}.", "",
            "| ROUGE F1 | Base | Fine-tuned | Différence |", "| --- | ---: | ---: | ---: |"]
        for kind in ("rouge1", "rouge2", "rougeL"):
            lines.append(f"| {kind} | {metrics.get('base_'+kind, float('nan')):.4f} | {metrics['finetuned_'+kind]:.4f} | {metrics.get('delta_'+kind, float('nan')):+.4f} |")
        lines.append("\nCes valeurs concernent uniquement les tailles et le split du run indiqué ; elles ne prouvent pas une amélioration sur le test complet.")
        if training:
            lines += ["", f"Durée d'entraînement mesurée par le trainer : {training.get('train_runtime', 'non disponible')} secondes, hors provisionnement et inférence. Loss de validation : {training.get('eval_loss', 'non disponible')}.",
                f"Paramètres entraînables : {training.get('trainable_parameters', 'non disponible')} sur {training.get('total_parameters', 'non disponible')}."]
        splits = sorted(frame["source_split"].unique())
        if len(splits) == 1 and (ROOT / f"reports/figures/restaurant_{splits[0]}_rouge.png").exists():
            lines += ["", f"![Comparaison base et LoRA sur le split indiqué](figures/restaurant_{splits[0]}_rouge.png)"]
        if (ROOT / "reports/figures/restaurant_training.png").exists():
            lines += ["", "![Loss pendant l'entraînement](figures/restaurant_training.png)"]
    else:
        lines += ["", "**Aucun résultat d'entraînement de qualité n'est encore téléchargé dans le dossier principal.** Les métriques du smoke test, s'il réussit, servent uniquement à valider la plomberie du workflow."]
    smoke_path = data / "smoke/run_summary.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        lines += ["", f"Essai technique téléchargé : `{smoke['job']}` ; état : `{smoke['state']}`. Les adaptateurs et scores de cet essai restent dans les dossiers smoke dédiés."]
    lines += ["", "## Test tenu à l'écart", ""]
    test_results = data / "final_test/evaluation_results.csv"
    test_summary = data / "final_test/run_summary.json"
    if test_results.exists() and test_summary.exists():
        import pandas as pd
        from agentic_workflows.restaurant_evaluation import score_restaurant_predictions

        test_run = json.loads(test_summary.read_text(encoding="utf-8"))
        test_frame = pd.read_csv(test_results, keep_default_na=False)
        if set(test_frame["source_split"]) != {"test"}:
            raise ValueError("The final-test folder must contain only the official test split.")
        _, test_metrics = score_restaurant_predictions(test_frame)
        lines += [f"Run : `{test_run['job']}`. Adaptateurs figés : `{test_run.get('parameters', {}).get('model_uri', 'voir provenance du run')}`.",
            f"{len(test_frame)} MR évalués sur {audit['splits']['test']['unique_mrs']} disponibles ; aucune nouvelle étape d'entraînement.", "",
            "| ROUGE F1 | Base | Fine-tuned | Différence |", "| --- | ---: | ---: | ---: |"]
        for kind in ("rouge1", "rouge2", "rougeL"):
            lines.append(f"| {kind} | {test_metrics['base_'+kind]:.4f} | {test_metrics['finetuned_'+kind]:.4f} | {test_metrics['delta_'+kind]:+.4f} |")
        if (ROOT / "reports/figures/restaurant_test_rouge.png").exists():
            lines += ["", "![Comparaison sur le test tenu à l'écart](figures/restaurant_test_rouge.png)"]
    else:
        lines.append("Aucun résultat du pipeline de test figé n'est encore téléchargé dans `data/restaurant/final_test/`.")
    browser_path = data / "browser_verification.json"
    lines += ["", "## Application et statut de la démonstration", ""]
    if browser_path.exists():
        browser = json.loads(browser_path.read_text(encoding="utf-8"))
        lines.append(f"Vérification navigateur : `{browser['mode']}` ; succès : `{browser['verified']}`.")
    else:
        lines.append("Aucune preuve de prédiction Restaurant via navigateur n'est encore enregistrée.")
    endpoint_path = data / "endpoint.json"
    if endpoint_path.exists():
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        lines.append(f"Endpoint Restaurant : `{endpoint['endpoint_resource']}` ; état local : `{endpoint['state']}`.")
    else:
        lines.append("Aucun endpoint Restaurant n'est enregistré localement ; les tests Yoda ne valident pas ce nouveau modèle.")
    registration_path = data / "registration.json"
    if registration_path.exists():
        registrations = json.loads(registration_path.read_text(encoding="utf-8"))["registrations"]
        for registration in registrations:
            lines.append(f"Modèle enregistré : `{registration['versioned_model_resource']}` ; source : `{registration['source_pipeline']}` ; lineage : `{registration.get('lineage', {}).get('status', 'non lié')}`.")
    demo_path = data / "live_demo.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        if demo.get("mode") != "live":
            raise ValueError("Live demo evidence must be explicitly marked as live.")
        lines += ["", "### Descriptions réellement servies", "", "| Exemple | Sortie |", "| --- | --- |"]
        for example, response in zip(demo["examples"], demo["responses"]):
            text = response.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {example['title']} | {text} |")
        if len(demo["responses"]) == 4:
            lines += ["", f"Même sortie pour le profil complet et sa permutation : `{demo['responses'][0] == demo['responses'][-1]}`."]
    if browser_path.exists() and browser.get("mode") == "live-prediction" and (ROOT / "reports/restaurant_chainlit.png").exists():
        lines += ["", "![Description Restaurant vérifiée dans Chainlit](restaurant_chainlit.png)"]
    lines += ["", "## Limites et présentation", "",
        "ROUGE mesure le recouvrement lexical. La couverture littérale des noms et lieux ne mesure pas toute la fidélité sémantique. Examiner aussi les omissions, ajouts et négations.",
        "Les références contiennent encore du bruit, et le pilote utilise par défaut un sous-ensemble déclaré. La base Phi-3 peut déjà avoir rencontré ce benchmark lors de son préentraînement ; on ne peut pas l'exclure.", "",
        "Voir le [protocole](../docs/restaurant_protocol.md) et le [déroulé de présentation](../docs/final_project_demo.md).",
        "Les anciens résultats [Session 3](session3_results.md) et [Session 4](session4_results.md) concernent Yoda.", ""]
    reports = ROOT / "reports"
    (reports / "restaurant_dataset_audit.json").write_text(json.dumps({"manifest": manifest, "preparation": audit}, indent=2), encoding="utf-8", newline="\n")
    path = reports / "restaurant_project.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
