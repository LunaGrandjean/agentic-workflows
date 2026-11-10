# Protocole expérimental

## Données et traçabilité

Dataset : [GEM/e2e_nlg](https://huggingface.co/datasets/GEM/e2e_nlg), version
GEM du corpus E2E nettoyé. Auteurs du corpus original : Jekaterina Novikova,
Ondřej Dušek et Verena Rieser. Source et nettoyage :
[tuetschek/e2e-cleaning](https://github.com/tuetschek/e2e-cleaning).
Les données sont annoncées sous licence **CC-BY-SA-4.0** ; les conversions de
données conservent cette attribution. Le code du dépôt est distinct du corpus.

Le script télécharge les fichiers Parquet publics sans exécuter l'ancien script
de chargement Hugging Face. Il enregistre les révisions source et Parquet, les
URL immuables et les SHA-256. Le composant Vertex vérifie les empreintes avant
de traiter les données. Aucun nouveau découpage aléatoire train/test n'est créé.

Les attributs sont normalisés dans un ordre fixe. Les répétitions d'un attribut
dans les données officielles sont conservées et comptées : certaines sont
contradictoires. L'interface refuse ces contradictions pour les nouveaux profils.
Les références peuvent elles-mêmes omettre ou ajouter des attributs ; une bonne
valeur de ROUGE ne garantit donc pas une description factuellement correcte.

L'audit reproductible `scripts/audit_restaurant_quality.py` trouve 1 443 lignes
train, 190 validation et 141 test avec plusieurs valeurs différentes pour un même
slot. Il trouve une référence textuelle identique entre train et validation
(casse/espaces normalisés), aucune entre train et test. Ces comptes ne sont pas
des chevauchements de MR. Le pilote conserve ces données ; filtrer les contradictions
du train serait une expérience supplémentaire, qui n'a pas été réalisée ici.

Les entrées identiques d'un même split d'évaluation sont regroupées, avec union
des références. Les variantes de cibles du train restent des exemples distincts.
Si un chevauchement canonique train/évaluation existe, seules les lignes train
concernées sont retirées et comptées. Un chevauchement validation/test provoque
un échec explicite. Sur le snapshot retenu, les trois chevauchements sont nuls.

## Entraînement et comparaison

- Base : `microsoft/Phi-3-mini-4k-instruct`, sans poids Yoda.
- QLoRA : NF4 4 bits, rang 8, alpha 16, dropout 0,05.
- Modules : `o_proj`, `qkv_proj`, `gate_up_proj`, `down_proj`.
- Configuration par défaut : 4 000 lignes train, 200 MR validation, 1 époque,
  learning rate 0,0001, batch 2, accumulation 8, longueur maximale 256.
- Les limites correspondent à un échantillonnage déterministe, seed 42. Ce
  pilote est une expérience sur un sous-ensemble, pas sur les 33 525 lignes.
- Le premier pilote réellement lancé utilise **1 000 lignes train** et 200 MR
  validation pour limiter le coût. Les paramètres et résultats effectifs du run
  figurent dans le rapport ; ils priment sur la configuration par défaut.
- La validation officielle sert à la loss et au choix du checkpoint. Pour
  calculer la loss, on utilise la première référence de chaque MR regroupé.
- Le smoke test utilise 32 lignes, 8 MR validation et 2 pas. Il vérifie seulement
  l'exécution ; ses métriques ne mesurent pas un modèle suffisamment entraîné.

L'inférence recharge la base en FP16 avec les adaptateurs. La comparaison de base
est obtenue sur les mêmes poids de base, avec `disable_adapter()`. Les deux
conditions reçoivent le même prompt et la même génération déterministe
(`do_sample=false`, `max_new_tokens=128`). Les cibles ne sont jamais incluses
dans le prompt d'inférence. Une limite de génération peut tronquer une sortie.

## Évaluation

ROUGE-1, ROUGE-2 et ROUGE-L F1, avec Porter stemming, sont calculés pour chaque
référence. Pour chaque MR et chaque métrique, le score retenu est le maximum
sur ses références ; le score global est la moyenne sur les MR uniques.
Ce choix est explicite et ne prétend pas reproduire tous les scores officiels
du challenge E2E. Les scores ne sont pas directement comparables au corpus Yoda.

On rapporte côte à côte la base, les adaptateurs et leur différence. On suit aussi
les sorties vides, la longueur et la présence littérale des noms et lieux fournis.
Cette dernière mesure est un diagnostic limité, pas un score complet de fidélité.

Les choix d'hyperparamètres et les essais utilisent la validation. Après choix du
modèle, `--evaluate-model gs://...` lance un pipeline dédié sur le test **sans
réentraîner**. Pour le bilan complet, fournir `--test-limit 0`. Si un sous-ensemble
est utilisé, indiquer sa taille réelle et ne pas le présenter comme le test complet.

Compléter par une lecture manuelle : fluidité, couverture des attributs, respect
des négations, absence d'informations ajoutées. Les exemples de démonstration
choisis à la main ne remplacent pas le résultat sur les données tenues à l'écart.
