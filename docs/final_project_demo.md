# Présentation du projet final : Restaurant Studio

Objectif : montrer la transformation d'un workflow de cours en une application
de génération de descriptions de restaurants. Les données et les sorties sont
en anglais ; la présentation peut être en français.

## Fil conducteur, environ 10 minutes

1. **Problème et exemple.** Montrer les caractéristiques dans
   `examples/restaurants.json`, puis une description. Le système doit exprimer
   les informations fournies sans inventer un prix ou une note manquante.
2. **Données.** GEM/e2e_nlg : 33 525 exemples train, 1 484 validation et
   1 847 test. Montrer le snapshot, ses empreintes et l'audit de chevauchement.
   Conserver les références alternatives pour l'évaluation.
3. **Workflow.** Montrer le graphe dans Vertex : préparation, entraînement QLoRA,
   inférence avec/sans adaptateurs, évaluation ROUGE. La validation officielle
   sert à suivre la loss ; le test reste réservé au bilan final.
4. **Modèle.** Phi-3-mini-4k-instruct, base chargée en 4 bits pour l'entraînement,
   adaptateurs LoRA. Expliquer que les adaptateurs sont sauvegardés séparément.
5. **Expérience supplémentaire.** Comparer la base et le modèle adapté avec
   exactement les mêmes prompts et paramètres. Ne pas présenter un smoke test
   comme une preuve de qualité.
6. **Résultat du test officiel.** Le modèle `restaurant-v1` a été évalué sur les
   1 847 exemples tenus à l'écart. ROUGE-1 passe de 0.5311 à 0.7386, ROUGE-2
   de 0.2901 à 0.4554 et ROUGE-L de 0.3858 à 0.5279. Une évaluation séparée de
   `restaurant-v2-security` sur les entrées acceptées par le parseur (1 706/1 847)
   donne, après garde-fou, ROUGE-1/2/L de 0.7292/0.4550/0.5479. Comparer sur le
   même sous-ensemble : `v1` adapté avait 0.7429/0.4620/0.5348. Le modèle est
   resté identique ; le prompt et les contrôles de serving ont changé.
7. **Application et sécurité.** Chainlit appelle le modèle Restaurant enregistré
   dans Vertex. Le parseur rejette des entrées invalides ou des tentatives
   d'injection ciblées avant l'appel au modèle. Le test officiel contenait 141
   entrées à attributs répétés, que l'app aurait rejetées ; le contrôle de sortie
   a remplacé 30 réponses par une description de secours. Montrer aussi les tests
   Langfuse ; ils ne couvrent pas le lot de 1 706 exemples.
8. **Limites, reproductibilité et coût.** ROUGE mesure le recouvrement lexical,
   pas la vérité de tous les faits. Les requêtes rejetées avant génération ne
   créent pas de score Langfuse. L'endpoint T4 coûte tant qu'il reste déployé.

## Préparer la démonstration

Depuis la racine du dépôt, utiliser `.venv-session5`. Vérifier d'abord les résultats
dans `reports/restaurant_project.md`. Les évaluations v1 et v2 ainsi qu'une
vérification Chainlit live ont réussi. Vérifier l'état Vertex avant la présentation,
car l'état local enregistré peut être périmé.

Prévoir le temps de déploiement de Vertex avant la présentation (le précédent
exercice a pris environ 20 minutes). Le déploiement utilise un T4 de serving,
dont le quota est distinct de celui de l'entraînement.

```powershell
.\.venv-session5\Scripts\python.exe scripts/manage_endpoint.py list
.\.venv-session5\Scripts\python.exe scripts/test_online_endpoint.py
.\.venv-session5\Scripts\python.exe scripts/run_chat.py
```

Si aucun endpoint Restaurant n'est déployé, le redéployer avant le test en direct :

```powershell
.\.venv-session5\Scripts\python.exe scripts/manage_endpoint.py deploy
```

Ouvrir `http://localhost:8000`. Dans un autre terminal :

```powershell
uvx --from playwright python scripts/verify_chainlit_browser.py
.\.venv-session5\Scripts\python.exe scripts/demo_restaurant.py
```

Ces commandes supposent qu'un **modèle Restaurant entraîné et enregistré** est
disponible. Les scripts ne réutilisent pas automatiquement le modèle Yoda.

## Les quatre exemples

Copier les profils de `examples/restaurants.json`. Comparer les descriptions :

- Profil complet : les attributs sont-ils tous exprimés ?
- Profil incomplet : le modèle invente-t-il un prix, une note ou un quartier ?
- Familles non acceptées : la négation est-elle respectée ?
- Même profil, ordre différent : le prétraitement produit le même prompt canonique.

Ajouter un exemple de sortie ratée si les résultats en contiennent. C'est utile
pour expliquer les limites plutôt que montrer uniquement les meilleurs cas.

## Solution de secours clairement annoncée

Si le réseau ou le quota de serving bloque la démo, montrer des sorties réellement
enregistrées lors du pipeline, en annonçant explicitement le mode différé :

```powershell
.\.venv-session5\Scripts\python.exe scripts/demo_restaurant.py --replay data/restaurant/final_test/evaluation_results.csv
```

Ce mode n'appelle pas Vertex et n'est pas une preuve d'inférence en direct. Garder
une capture du graphe terminé, les résultats et la provenance du run choisi.

## Après la démonstration

Arrêter Chainlit avec Ctrl+C, puis :

```powershell
.\.venv-session5\Scripts\python.exe scripts/manage_endpoint.py cleanup
.\.venv-session5\Scripts\python.exe scripts/manage_endpoint.py list
```

La suppression concerne l'endpoint temporaire du projet Restaurant. Les versions
enregistrées, les poids dans GCS et les anciens résultats Yoda sont conservés.
