# Projet final — Restaurant Studio

État documenté le 10/11/2026. Les résultats ci-dessous proviennent des fichiers réellement produits.

## Données vérifiées

Dataset : [GEM/e2e_nlg](https://huggingface.co/datasets/GEM/e2e_nlg), anglais, licence CC-BY-SA-4.0.
Révision source : `0e089c2ba61c3a0d183815c87c2c95e98fb446a6`. Révision Parquet : `f97230c6d02c1678ec0279be4dea6dc722803a0a`.

| Split officiel | Lignes source | MR uniques | Lignes préparées, corpus complet |
| --- | ---: | ---: | ---: |
| train | 33525 | 12568 | 33525 |
| validation | 1484 | 1484 | 1484 |
| test | 1847 | 1847 | 1847 |

MR avec valeurs contradictoires pour un même slot : `{"train": 1443, "validation": 190, "test": 141}`.
Références textuelles identiques entre splits, après normalisation casse/espaces : `{"train_validation": 1, "train_test": 0, "validation_test": 0}`.
Ces diagnostics sont distincts des chevauchements d'entrées ci-dessous. Le pilote conserve le corpus officiel ; aucun filtrage des contradictions n'a été expérimenté.

Chevauchements après normalisation : `{"train_validation": 0, "train_test": 0, "validation_test": 0}`.
Lignes train retirées pour chevauchement : 0.
Les répétitions de slots dans le corpus sont conservées et signalées ; elles ne sont pas silencieusement écrasées.

## Ce qui est adapté

Préparation reproductible, splits officiels, prompt Restaurant commun au train et au serving, validation dédiée, comparaison base/LoRA, ROUGE multi-références, registre et endpoints séparés de Yoda, application Chainlit et exemples de démonstration.

Le pipeline de test dédié évalue des adaptateurs figés sans réentraîner. Le code des sessions Yoda reste disponible.

## Exécutions et résultats

- Run soumis : `projects/123905721265/locations/europe-west2/pipelineJobs/lulik-restaurant-description-training-20261110175124` ; smoke test : `True` ; dernier état enregistré : `PIPELINE_STATE_SUCCEEDED`.
- Run soumis : `projects/123905721265/locations/europe-west2/pipelineJobs/lulik-restaurant-description-training-20261110181012` ; smoke test : `False` ; dernier état enregistré : `PIPELINE_STATE_SUCCEEDED`.

Run téléchargé : `projects/123905721265/locations/europe-west2/pipelineJobs/lulik-restaurant-description-training-20261110181012` ; état : `PIPELINE_STATE_SUCCEEDED`.
Taille train demandée : 1000.0 ; validation : 200.0 ; époques configurées : 1.0 ; learning rate : 0.0001.
Évaluation sur 200 MR ; split(s) : validation.

| ROUGE F1 | Base | Fine-tuned | Différence |
| --- | ---: | ---: | ---: |
| rouge1 | 0.5056 | 0.7358 | +0.2302 |
| rouge2 | 0.2801 | 0.4713 | +0.1912 |
| rougeL | 0.3826 | 0.5555 | +0.1728 |

Ces valeurs concernent uniquement les tailles et le split du run indiqué ; elles ne prouvent pas une amélioration sur le test complet.

Durée d'entraînement mesurée par le trainer : 473.0634 secondes, hors provisionnement et inférence. Loss de validation : 0.6234755516052246.
Paramètres entraînables : 12582912.0 sur 3833662464.0.

![Comparaison base et LoRA sur le split indiqué](figures/restaurant_validation_rouge.png)

![Loss pendant l'entraînement](figures/restaurant_training.png)

Essai technique téléchargé : `projects/123905721265/locations/europe-west2/pipelineJobs/lulik-restaurant-description-training-20261110175124` ; état : `PIPELINE_STATE_SUCCEEDED`. Les adaptateurs et scores de cet essai restent dans les dossiers smoke dédiés.

## Test tenu à l'écart

Aucun résultat du pipeline de test figé n'est encore téléchargé dans `data/restaurant/final_test/`.

## Application et statut de la démonstration

Vérification navigateur : `offline-error-flow` ; succès : `True`.
Aucun endpoint Restaurant n'est enregistré localement ; les tests Yoda ne valident pas ce nouveau modèle.

## Limites et présentation

ROUGE mesure le recouvrement lexical. La couverture littérale des noms et lieux ne mesure pas toute la fidélité sémantique. Examiner aussi les omissions, ajouts et négations.
Les références contiennent encore du bruit, et le pilote utilise par défaut un sous-ensemble déclaré. La base Phi-3 peut déjà avoir rencontré ce benchmark lors de son préentraînement ; on ne peut pas l'exclure.

Voir le [protocole](../docs/restaurant_protocol.md) et le [déroulé de présentation](../docs/final_project_demo.md).
Les anciens résultats [Session 3](session3_results.md) et [Session 4](session4_results.md) concernent Yoda.
