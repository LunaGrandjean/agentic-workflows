# Session 3 results — 22 October 2026

The four-stage pipeline is implemented. Both the smoke test and full run completed
successfully on Vertex AI: data preparation, Phi-3 LoRA fine-tuning, inference and
Ragas evaluation. Seven local tests pass. Both the Session 2 and Session 3 pipelines compile.

## Verified smoke test

[Vertex run](https://console.cloud.google.com/vertex-ai/locations/europe-west2/pipelines/runs/lulik-yoda-model-training-20261022195529?authuser=1&project=agentics-workflows-510809)

- Dataset: 720 sentences; 576 training rows and 144 held-out test rows.
- Fine-tuning validation: 58 rows from the training split, leaving 518 training rows.
- LoRA: rank 8, alpha 16, dropout 0.05, targeting `qkv_proj`, `o_proj`,
  `gate_up_proj` and `down_proj`.
- Trainable parameters: 12,582,912 / 3,833,662,464, or **0.328%**.
- Adapter inspection: 256 tensors with the expected target modules and rank.
- Smoke training: two optimizer steps; training loss 2.9138; validation loss 2.7958.
- Predictions: three test rows, with the expected CSV columns.
- Mean ROUGE F1: ROUGE-1 0.4510, ROUGE-2 0.2840, ROUGE-L 0.3320.

Recomputing the downloaded predictions locally reproduced the cloud ROUGE metrics.
TensorBoard successfully parsed both training and validation loss events at steps 1–2.
These two training steps validate execution; their mostly ordinary-English predictions
do not establish that the model has learned the Yoda style.

Downloaded smoke artifacts are in `data/session3_smoke/`,
`models/phi3-yoda-smoke-adapter/` and `logs/session3_smoke/`. These directories are
excluded from Git. The per-run JSON summary preserves artifact URIs and cloud metrics.

## Full run

[Vertex run](https://console.cloud.google.com/vertex-ai/locations/europe-west2/pipelines/runs/lulik-yoda-model-training-20261022220009?authuser=1&project=agentics-workflows-510809)

Submitted with three training epochs, learning rate 0.0003, maximum sequence length
128, micro-batch size 4 and gradient accumulation over 4 batches. Inference is configured
for all 144 held-out rows, with at most 64 generated tokens per row.

**Final state: `PIPELINE_STATE_SUCCEEDED`; all four tasks succeeded.**

Training completed 96 optimizer steps (reported epoch 2.954 for the configured three
epochs) in 354.5 seconds. The saved final adapter was reloaded successfully by the
GPU inference task and evaluated on all 144 held-out
test rows. There were no empty responses. Training and test input sentences do not overlap.

| Metric | Full-run value |
| --- | ---: |
| Mean training loss | 1.2741 |
| Final validation loss | 1.3832 |
| Test ROUGE-1 F1 | 0.9741 |
| Test ROUGE-2 F1 | 0.8551 |
| Test ROUGE-L F1 | 0.8856 |
| Test predictions | 144 |

The downloaded per-sample scores reproduce the aggregated metrics saved by Vertex.

### Loss evolution

| Approximate epoch | Optimizer step | Validation loss |
| --- | ---: | ---: |
| 1 | 32 | 1.3336 |
| 2 | 65 | 1.3123 |
| 3 | 96 | 1.3832 |

The logged training loss falls from 3.0516 at step 1 to 0.9144 at step 96.
Validation loss improves between the first and second epoch, then increases slightly
at the final checkpoint. This suggests some overfitting late in training; the final
adapter is not the checkpoint with the lowest observed validation loss. Selecting the
best validation checkpoint or trying two epochs would be useful follow-up experiments.
The smoke run's validation loss was 2.7958 after only two optimizer steps.

### Example predictions

| Input | Reference | Fine-tuned response |
| --- | --- | --- |
| He wrote down a long list of items. | A long list of items, down he wrote. | Down a long list of items, he wrote. |
| The duke left the park in a silver coach. | Left the park in a silver coach, the duke did. | In a silver coach, the duke left the park. |
| A siege will crack the strong defense. | Crack the strong defense, a siege will. | Crack the strong defense, a siege will. |

These examples show that the fine-tuned model has learned the requested word-order
changes, although outputs are not always identical to the reference or equally natural.
ROUGE-1 is especially high because this task largely preserves the source vocabulary;
the scores measure lexical overlap rather than fully judging Yoda style or grammatical quality.

### Downloaded artifacts and checks

- `data/session3/predictions.csv`: all 144 predictions in the expected format.
- `data/session3/evaluation_results.csv`: predictions with per-sample ROUGE scores.
- `data/session3/run_summary.json`: successful state, artifact URIs and aggregated cloud metrics.
- `models/phi3-yoda-adapter/`: adapter weights, tokenizer, hyperparameters and training history.
- `logs/session3/`: two TensorBoard event files, successfully parsed with training and
  validation tags through optimizer step 96, without out-of-order steps.

These downloaded artifact directories are excluded from Git. GPU training and model
reloading/generation were verified on Vertex; the notebook provides equivalent local
experiments for a CUDA-capable machine. No local GPU training or generation was required
to validate the cloud pipeline.

Open the downloaded training curves with:

```powershell
uvx --from "tensorboard>=2.18,<3" python -m tensorboard.main --logdir logs/session3
```

Then open `http://localhost:6006`. The implementation and full cloud execution are complete;
the artifacts, sample predictions and training/evaluation results have been inspected.
