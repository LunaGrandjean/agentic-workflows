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
   exactement les mêmes prompts et paramètres. Montrer les différences et les
   cas où le fine-tuning ne gagne pas. Ne pas présenter un smoke test comme une
   preuve de qualité ; indiquer les tailles réellement utilisées.
6. **Application en direct.** Chainlit applique le même prompt qu'à l'entraînement,
   authentifie la requête HTTP et appelle le modèle enregistré dans Vertex.
7. **Robustesse et limites.** Montrer l'ordre des attributs permuté, un profil
   incomplet et `familyFriendly[no]`. Examiner manuellement omissions et ajouts.
   ROUGE mesure le recouvrement lexical, pas la vérité de tous les faits.
8. **Reproductibilité et coût.** Montrer les versions du modèle, sa provenance,
   les métriques enregistrées et la commande de suppression de l'endpoint.

## Préparer la démonstration

Depuis la racine du dépôt, utiliser `.venv-session4`. Vérifier d'abord les résultats
dans `reports/restaurant_project.md` : un statut « local » ou « smoke » ne signifie
pas que l'entraînement de qualité et la démo live ont réussi.

Prévoir le temps de déploiement de Vertex avant la présentation (le précédent
exercice a pris environ 20 minutes). Le déploiement utilise un T4 de serving,
dont le quota est distinct de celui de l'entraînement.

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py deploy
.\.venv-session4\Scripts\python.exe scripts/test_online_endpoint.py
.\.venv-session4\Scripts\python.exe scripts/run_chat.py
```

Ouvrir `http://localhost:8000`. Dans un autre terminal :

```powershell
uvx --from playwright python scripts/verify_chainlit_browser.py
.\.venv-session4\Scripts\python.exe scripts/demo_restaurant.py
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
.\.venv-session4\Scripts\python.exe scripts/demo_restaurant.py --replay data/restaurant/evaluation_results.csv
```

Ce mode n'appelle pas Vertex et n'est pas une preuve d'inférence en direct. Garder
une capture du graphe terminé, les résultats et la provenance du run choisi.

## Après la démonstration

Arrêter Chainlit avec Ctrl+C, puis :

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py cleanup
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py list
```

La suppression concerne l'endpoint temporaire du projet Restaurant. Les versions
enregistrées, les poids dans GCS et les anciens résultats Yoda sont conservés.
