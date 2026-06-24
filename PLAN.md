# Staj-I Execution Plan: Plant Disease Classification

## Phase Breakdown

### Phase 0 — Environment Verification
**Purpose:** Ensure the local setup is stable before any model or data work begins.
**Checklist:**
- Confirm GPU access is working (`torch.cuda.is_available()` should be `True`) and available VRAM is at least 4GB.
- Confirm required project folders exist (`data/`, `src/`, `notebooks/`).
- Confirm `requirements.txt` is present and tracked.
- Run a minimal CUDA sanity check (`torch.randn(10,10).cuda()`) without errors.
**Failure risks:** Driver/CUDA mismatches or environment drift can interrupt later training phases.
**Completion rule:** Once all checks pass, freeze environment changes and proceed to Phase 1 without further tuning.

---

### Phase 1 — Data Pipeline
**Days:** 2–4
**Objective:** PlantVillage loads, splits, and batches correctly. Nothing about modeling yet.
**Deliverables:**
- `src/dataset.py`: `ImageFolder`-based loader, transform pipeline (`Resize`, `ToTensor`, `Normalize` with ImageNet stats since you're using a pretrained backbone)
- Train/val/test split (80/10/10), with split logic seeded and reproducible
- One notebook cell that visualizes a batch (images + labels) to confirm correctness by eye
**Must NOT optimize:** Data augmentation sophistication. Use `Resize` + `ToTensor` + `Normalize` only. Augmentation is a Phase 3 refinement, not a Phase 1 requirement.
**Failure risks:** Silent label mismatches (folder names don't map to what you expect — verify with `dataset.classes`). Forgetting `Normalize` and wondering why a pretrained backbone underperforms.
**Exit condition:** You can pull one batch, print its shape, and print the corresponding class names correctly. That's it.

---

### Phase 2 — Rehearsal Training Loop
**Days:** 5–6
**Objective:** Validate the training loop mechanics on a *small, fast* subset — disposable code, not a checkpoint to keep.
**Deliverables:**
- `src/train.py`: forward pass, loss computation, backward pass, optimizer step, per-epoch loss printout
- Run on either CIFAR-10 or a 3–4 class subset of PlantVillage, 3–5 epochs
- Confirm loss decreases — that's the only success metric here
**Must NOT optimize:** Accuracy, architecture choice, hyperparameters. If loss goes down at all, the phase is successful.
**Failure risks:** Treating this rehearsal model as precious and over-investing time tuning it. Discard it once the loop is proven.
**Exit condition:** Loss curve trends downward across epochs with no crashes. Delete or archive this checkpoint — it is not your deliverable.

---

### Phase 3 — Full PlantVillage Training
**Days:** 7–11
**Objective:** Train the actual model on the full 38-class dataset.
**Deliverables:**
- `src/model.py`: ResNet18 (`weights="IMAGENET1K_V1"`), final `fc` layer replaced for 15 classes
- Full training run, checkpointed (`torch.save`) at best validation accuracy
- Training/validation loss and accuracy curves logged (simple CSV or matplotlib plot — no MLflow needed at this scope)
- Batch size tuned to your 4GB VRAM ceiling — start at 16, increase only if `nvidia-smi` shows headroom
**GPU constraint handling:** If you hit CUDA OOM, reduce batch size before reducing image resolution. If still OOM, switch this run to a Kaggle T4 notebook — don't lose days fighting local memory limits.
**Must NOT optimize:** Squeezing out the last 1–2% accuracy via architecture search or exotic augmentation. A solid ResNet18 baseline is the internship-appropriate deliverable, not a leaderboard submission.
**Failure risks:** Training silently overfitting (train accuracy ~100%, val accuracy stagnant) — watch the curves, don't just wait for completion.
**Exit condition:** Validation accuracy plateaus and you have a saved checkpoint you trust. Stop here even if accuracy could theoretically go higher with more tuning.

---

### Phase 4 — Evaluation on PlantVillage
**Days:** 12–13
**Objective:** Honest, complete metrics on the held-out PlantVillage test set.
**Deliverables:**
- `src/evaluate.py`: accuracy, per-class precision/recall/F1, confusion matrix
- One paragraph identifying which classes get confused and a plausible reason (visual similarity, class imbalance)
**Must NOT optimize:** Don't retrain based on test set performance — that invalidates the test set. If results are concerning, that's a finding to document, not a reason to keep tuning.
**Failure risks:** Reporting only overall accuracy and missing that one or two classes are doing badly while masking it in the aggregate number — this is the single most common shallow-analysis mistake in these projects.
**Exit condition:** A clear written summary with numbers, not just "the model works well."

---

### Phase 5 — PlantDoc Generalization Check
**Days:** 14–17
**Objective:** Measure and partially close the lab-to-real-world gap. This is the phase that differentiates your project from a generic notebook.
**Deliverables:**
- Step 1 (Day 14): Run the PlantVillage-trained model, untouched, on PlantDoc's test split. Record accuracy — expect a meaningful drop.
- Step 2 (Day 15–16): Fine-tune the same checkpoint on PlantDoc's small training split (54–137 images/class) for a small number of epochs (5–10, monitor for overfitting given the small data size).
- Step 3 (Day 17): Re-evaluate on PlantDoc test split. Report before/after numbers.
**Must NOT optimize:** Don't try to make PlantDoc accuracy match PlantVillage accuracy — that's not realistic given PlantDoc's size and difficulty, and chasing it risks overfitting on a tiny dataset. The *gap and its partial closure* is the result, not parity.
**Failure risks:** Fine-tuning on 54–137 images/class for too many epochs causes immediate overfitting — watch validation loss closely, stop early.
**Exit condition:** You have two numbers (before/after fine-tuning) and one sentence explaining what they mean.

---

### Phase 6 — Documentation and Cleanup
**Days:** 18–19
**Objective:** Repository is presentable to a stranger.
**Deliverables:**
- README rewritten (see structure below)
- Dead code, throwaway notebooks, and Phase 2 rehearsal artifacts removed or archived into a clearly labeled `experiments/` folder
- Final results (Phase 4 + Phase 5 numbers) consolidated into one results section
**Exit condition:** Someone unfamiliar with the project can clone the repo and understand what it does within 10 minutes from the README alone.

---

### Phase 7 — Buffer / Presentation Prep
**Days:** 20
**Objective:** Reserved slack. Do not schedule new technical work here.
**Deliverables:** Whatever internal report/presentation Staj-I requires, built from Phase 6 documentation.
**Exit condition:** N/A — this day exists because Phases 0–6 will slip somewhere, and this absorbs it without breaking your end date.

---

## Engineering Discipline Constraints

**What must NOT be optimized:**
- Architecture novelty — ResNet18 is correct and sufficient. Do not switch architectures mid-project chasing marginal gains.
- Accuracy beyond a reasonable plateau — diminishing returns here cost days you don't have.
- Data augmentation sophistication — basic resize/normalize is enough at this scope.
- PlantDoc-to-PlantVillage parity — not achievable or expected given dataset size disparity.

**What should be prioritized:**
- Pipeline reproducibility — anyone (including future you) can rerun any script and get consistent results
- Metrics clarity — per-class breakdown, not just aggregate accuracy
- Honest reporting of the generalization gap — this is the actual engineering insight of the project
- Clean separation between experimentation and final code

**Common failure points and prevention:**
- *Environment drift mid-project* → don't touch driver/CUDA setup again once Phase 0 passes; if something breaks, isolate the fix to a separate session, don't let it bleed into training days
- *Treating rehearsal model as final* → explicitly archive or delete Phase 2 outputs
- *Silent overfitting* → always plot train vs. val curves, never trust a single final-epoch number
- *Scope creep into deployment* → serving/API is Staj-II. Resist building it early even if it feels like progress.

**Definition of "done" per phase:** stated explicitly in each Exit Condition above — each is a binary, checkable condition, not a feeling of "good enough."

---

## Repository and Workflow Architecture

**Where to maintain the day-by-day plan:**

Put it in the repo itself, not Notion, not a separate doc. Create `PLAN.md` at the repo root with the phase table above, and check off exit conditions as you hit them. Reasons: it stays version-controlled alongside the code it governs, your mentor can see it if you share the repo, and it removes any friction of switching tools to track progress.

**Repository structure:**

```
crop-disease-detector/
├── PLAN.md                  # the phase breakdown above, checked off as you go
├── README.md                # final project description (written in Phase 6, drafted earlier)
├── requirements.txt
├── data/                     # gitignored — never commit datasets
├── notebooks/
│   └── experiments/          # exploratory work, visualization, rehearsal — messy is fine here
├── src/
│   ├── dataset.py             # ImageFolder loading, transforms, splits
│   ├── model.py                # ResNet18 setup, fc layer replacement
│   ├── train.py                 # training loop, checkpointing
│   ├── evaluate.py              # metrics, confusion matrix
│   └── finetune_plantdoc.py     # Phase 5 fine-tuning script, separate from main train.py
├── checkpoints/              # gitignored — saved model weights
└── results/
    ├── plantvillage_metrics.json
    └── plantdoc_before_after.json
```

**Notebooks vs. src/ — the rule:**

Notebooks are for exploration only: visualizing a batch, checking class distributions, plotting a confusion matrix once you have the numbers. The moment code needs to be run twice or reused (training loop, evaluation logic, data loading), it moves into `src/` as a proper function or script, callable from the command line (`python src/train.py`). If you find yourself copy-pasting a notebook cell into another notebook, that's the signal it belongs in `src/`.

**Reproducibility for training/evaluation scripts:**

Every script in `src/` should run standalone via `python src/script_name.py` with arguments (use `argparse`, even minimally — `--epochs`, `--batch-size`) rather than hardcoded values buried in a notebook. This is what makes Phase 6 documentation trivial: the README just says "run `python src/train.py --epochs 15`" instead of "open this notebook and run cells 1 through 47 in order."

**.gitignore essentials:**
```
data/
checkpoints/
venv/
__pycache__/
*.ipynb_checkpoints/
```

Never commit the dataset or model weights to git — both are large and regenerable. Commit code, configs, and results (small JSON/CSV metric files) only.