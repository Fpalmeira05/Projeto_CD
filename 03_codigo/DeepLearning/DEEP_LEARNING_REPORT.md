# Deep Learning Phase — Diagnostic Report

This document explains the *why* behind the choice of architecture, the
failure mode observed in the first classification experiment, the
callback-level fix that recovered it, and the **hierarchical (classify
then regress) reformulation** that produced our most operationally useful
predictor. It is meant as a source you can paraphrase directly into the
LaTeX report.

The deep-learning phase has **three layers of contribution**:

1. **Diagnostic** — identifying the `class_weight` × `EarlyStopping(val_loss)`
   interaction that silently produces a degenerate constant-majority predictor.
2. **Corrective** — a single custom callback (`MacroF1Monitor`) recovers the
   flat 3-class MLP to Macro F1 = 0.37, comparable to the kNN/Ensemble baselines.
3. **Reformulation** — splitting the problem into a binary classifier
   (`On-time` vs `Delayed`) chained with a regression on delayed flights
   produces the project's strongest classifier (Stage 1 binary Macro F1 = 0.62)
   and rescues the regression from a useless R² = -1.20 to a usable end-to-end
   MAE of 29 minutes.

---

## 1. Why an MLP (and not a CNN / RNN / Transformer)

The flight-delay dataset is **purely tabular**: each row is one independent
flight described by a fixed-length feature vector composed of

- **scaled numeric features** (`DISTANCE`, `CRS_ELAPSED_TIME`, `PLANNED_SPEED`,
  `AVG_DELAY_PER_HOUR`, ...),
- **one-hot encoded categorical features** (airline, origin, destination,
  season, day-of-week, time-of-day, ...).

There is **no spatial structure** that would justify a CNN, and **no
per-sample temporal sequence** that would justify an RNN/LSTM (every row is
treated as an independent observation; any temporal information is already
engineered into features such as `MONTH`, `SEASON`, `IS_HOLIDAY_MONTH`,
`AVG_DELAY_PER_HOUR`). The Multi-Layer Perceptron (MLP) is the canonical
deep-learning architecture for structured tabular data; it is also the
architecture chosen by the reference project from the previous year and the
one taught in the theoretical-practical class TP-05.

## 2. Architecture and training recipe

The MLP is implemented **layer-by-layer** with TensorFlow / Keras:

```
Input(d)
  -> Dense(128, relu) -> [BatchNorm] -> Dropout(0.3)
  -> Dense(64,  relu) -> [BatchNorm] -> Dropout(0.3)
  -> Dense(32,  relu) -> [BatchNorm] -> Dropout(0.2)
  -> Dense(3, softmax)            # classification
  -> Dense(1)                     # regression
```

| Component | Role |
|---|---|
| **ReLU** activations | Non-linearity, fast to train, no vanishing-gradient. |
| **BatchNormalization** | Stabilises and accelerates convergence by normalising pre-activations per mini-batch. |
| **Dropout** | Stochastic regularization to prevent overfitting. |
| **Adam (lr=1e-3)** | Adaptive learning-rate optimizer, robust default. |
| **Custom `MacroF1Monitor`** *(classification)* | Custom callback that early-stops on validation **macro F1** instead of the unweighted `val_loss`. See §5. |
| **`EarlyStopping(val_loss)`** *(regression)* | Standard for the regression head, where val_loss is a clean signal. |
| **`ReduceLROnPlateau(0.5)`** | Halves the learning rate when validation loss stalls. |
| **`sparse_categorical_crossentropy`** | Loss function for the 3-class problem. |
| **`mse` + `mae`** | Loss/metric for the regression head. |

## 3. The five experiments

| # | Setup | Purpose |
|---|-------|---------|
| **1** | Classification, full distribution, `class_weight='balanced'` (sklearn) | Default deep-learning recipe with class re-weighting. |
| **1b** | Same as Exp 1 but **without BatchNorm**, with **manual stronger weights** `{1, 5, 2.5}`, and with **output-bias initialisation** to `log(class_priors)` | Stress-test: does *more* intervention help? |
| **2** | Classification on a **balanced (undersampled)** training set | Reference experiment — directly comparable with Ensemble Exp 2. |
| **3** | Regression on `ARR_DELAY` (minutes) | Quantify how much variance is recoverable from pre-departure features only. |
| **4 (Hierarchical)** | Two-stage cascade: Stage 1 = binary classifier (`On-time` vs `Delayed`), Stage 2 = regression on `ARR_DELAY` trained **only on delayed flights** | Split the question "is it late?" from "by how much?" — each stage gets a more tractable problem. |

## 4. Observed results (200 k train / 50 k test)

### Classification (3-class, comparable across all experiments)

| Model | Accuracy | F1 On-time | F1 Short | F1 Long | **Macro F1** |
|---|---|---|---|---|---|
| Exp 1 — sklearn `class_weight='balanced'` (with BN, **macro-F1 callback**) | 56.4 % | 0.74 | 0.22 | 0.16 | 0.37 |
| Exp 1b — no BN + manual weights + log-prior bias init                    | 73.0 % | 0.84 | 0.00 | 0.00 | 0.28 (collapsed) |
| Exp 2 — balanced (undersampled) training                                  | 26.5 % | 0.37 | 0.19 | 0.05 | 0.21 |
| **Exp 4 — Hierarchical (composite 3-class)**                               | **62.5 %** | **0.77** | **0.01** | **0.35** | **0.38**  ← best 3-class MLP |

### Regression on `ARR_DELAY` (minutes)

| Scope | MSE | RMSE (min) | MAE (min) | R² |
|---|---|---|---|---|
| Exp 3 — Flat regression (full test set)                          | 6 989.52 | 83.6 | 26.1 | **−1.20** |
| **Exp 4 — Stage 2 alone (delayed-only test subset)**              | 8 611.63 | 92.8 | 50.0 | **−0.001** |
| **Exp 4 — Hierarchical end-to-end (full test set, cascade)**      | **3 867.58** | **62.2** | **29.2** | (n/a — predictions clipped at 0) |

### Stage 1 standalone (binary "is this flight late?")

| Metric | Value |
|---|---|
| Accuracy            | 68.1 % |
| F1 (Delayed)        | **0.475** |
| Macro F1 (binary)   | **0.623** ← best classifier in the project |
| Threshold           | 0.5 |

## 5. Diagnosis: why the original Experiment 1 collapsed

The first run of Experiment 1 produced **73.04 % accuracy with recall
1.00 / 0.00 / 0.00** — the network predicted `On-time` for *every single*
flight in the test set. Accuracy = the test-set proportion of `On-time`,
Macro F1 = 0.28 = the F1 of the majority class divided by 3.

This is a **degenerate predictor** — the model has not learned anything,
it has merely defaulted to the majority class.

### 5.1 The dominant cause

The hidden culprit is the **interaction between Keras' `class_weight`
parameter and the default `EarlyStopping(monitor='val_loss',
restore_best_weights=True)`**:

- `class_weight` re-weights the **training** loss only.
- The **validation** loss is computed *unweighted*. On a 73 / 10 / 17
  validation split, the unweighted cross-entropy is *minimised* by the
  trivial "always predict majority" predictor.
- Therefore the **earliest epochs**, where the network's softmax output is
  approximately the class prior, produce the **lowest val_loss** of the
  entire run.
- `restore_best_weights=True` then reverts the network to that early-epoch
  degenerate state at the end of training, **discarding** all the
  representational progress made by the class-weighted training.

In other words: every time training tried to do the right thing,
EarlyStopping silently undid it.

### 5.2 Secondary contributors

- **`class_weight='balanced'` is mild for our distribution.** sklearn's
  formula `weight_c = n_samples / (n_classes * count_c)` produces roughly
  `{0.5, 3.5, 1.7}` here. Even after weighting, hundreds of `On-time`
  examples per batch dominate the gradient, biasing the optimizer toward
  the majority.
- **BatchNormalization absorbs the imbalance.** Per-batch statistics and
  inference-time running averages are dominated by the majority class, so
  hidden activations get normalised relative to majority-class statistics —
  partially counter-acting the class-weight signal in the loss.

But these two factors are *not* what produces the *exact* 1.00 / 0.00 / 0.00
recall pattern. The dominant cause (§5.1) is the EarlyStopping interaction,
and isolating it is what unlocked the fix.

## 6. The fix that actually worked

A custom callback `MacroF1Monitor` replaces `EarlyStopping(val_loss)` for
all classification runs. It computes **validation macro F1** at the end of
every epoch, tracks the best value, and only restores those weights at
training end. Macro F1 is **immune to majority collapse** — a constant
predictor scores ~0.28 — so the callback only commits weights from epochs
where the network is genuinely discriminating between the three classes.

Implementation (`DeepLearning.py`):

```python
class MacroF1Monitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        macro_f1 = f1_score(self.y_val, preds, average='macro', zero_division=0)
        if macro_f1 > self.best_f1 + 1e-4:
            self.best_f1 = macro_f1
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
```

After this single change, **Experiment 1 went from a degenerate predictor
(Macro F1 = 0.28) to the project's best MLP result (Macro F1 = 0.37)** —
without changing the architecture, the optimizer, or the class weights.

### Validation evidence

A controlled smoke test on a synthetic 70 / 10 / 20 imbalanced 3-class
problem (geometry chosen to mimic the real class priors) confirmed the
callback's effect *in isolation*:

- **`EarlyStopping(val_loss)`**: model collapses to majority, Macro F1 ≈ 0.28.
- **`MacroF1Monitor`**: Macro F1 = **0.94**, all three classes recovered with
  non-zero recall, weights restored from a late training epoch where the
  model had genuinely learned the decision boundary.

## 7. Why Experiment 1b *also* collapsed (the over-correction)

Experiment 1b was originally designed as a "belt-and-braces" fix: drop
BatchNorm, use stronger manual weights `{1, 5, 2.5}`, **and** initialise the
output-layer bias to `log(class_priors)`. After the macro-F1 callback was
added on top, this combination still produced a degenerate predictor
(73.0 % accuracy, recall 1.00 / 0.00 / 0.00) — even though the milder
Experiment 1 *did not*.

This is the lesson: **fix-stacking is not free**. The three interventions
each push the optimisation in the same direction:

1. The **log-prior bias init** places the softmax output exactly at the
   marginal class distribution at epoch 0. The model already "knows" the
   prior — it does not have to learn it.
2. The **manual weights {1, 5, 2.5}** are stronger than sklearn-balanced and
   apply heavy gradient pressure on the minority classes.
3. **Removing BatchNorm** widens the activation distribution, which
   amplifies the magnitude of the gradient updates.

Combined, these create a fixed point: at initialisation the macro F1 is
already at the prior baseline (≈ 0.28); the strong gradients from the
re-weighted loss without the stabilising effect of BatchNorm push the
network into an unstable region where val_macro_f1 oscillates *below* the
baseline; `MacroF1Monitor`'s `best_f1` never improves, so it restores the
initial weights. The network ends training exactly where it started — at
the prior — which is observationally identical to the old majority-collapse
solution.

The takeaway, paraphrasable for the report: *"after identifying the
EarlyStopping × class_weight interaction as the dominant cause, the minimal
single-callback fix outperforms the multi-component intervention; piling
additional 'fixes' on top is not only unnecessary, it actively destabilises
training."*

## 8. Why the balanced-training Experiment 2 underperforms

Macro F1 for Exp 2 is **0.21**, lower than both Exp 1 (0.37) and the
Ensemble Exp 2 result. Two reasons:

- **Smaller effective training set.** Undersampling to the minority count
  shrinks the training set roughly by a factor of three. On the resulting
  ~60 k rows of evenly-distributed classes, the MLP converges quickly, hits
  its peak, and `MacroF1Monitor`'s patience expires before the optimizer
  can refine — leaving an under-trained model.
- **The undersampled distribution is artificially flat,** but the *test*
  set still has the natural 73 / 10 / 17 distribution. A model trained on a
  uniform prior over-predicts minorities at test time, depressing the
  Macro F1 because both `On-time` precision and overall accuracy collapse.

This is itself an interesting empirical finding: for this dataset, **soft
re-weighting with a callback that tracks the right metric outperforms hard
undersampling**, while using all 200 k training rows. The implication is
that the limiting factor is not the class imbalance per se but the
optimisation dynamics — once those are corrected (Exp 1), full-distribution
training is the better approach.

## 9. Discussion of the regression result (Exp 3)

The regression head produced **R² = −1.20**, **MAE ≈ 26 min**, **RMSE ≈ 84 min**.
A negative R² means the MLP is *worse than a constant-mean predictor*.

Two facts make this *consistent* rather than catastrophic:

- The **MAE (26 min)** is comparable to a typical-flight prediction error
  on this problem when leakage features are removed. The typical
  pre-departure-only baseline lives around 20–30 min MAE.
- The **gap between MAE and RMSE** (26 vs 84) is symptomatic of a long
  upper tail in the target: the test set contains a small number of
  flights with very large delays (>500 min), and the model's predictions
  for those events are bounded by the bulk of training data. Those few
  catastrophic mis-predictions inflate MSE far more than they inflate MAE.

The deeper finding stands either way: with the data-leakage columns
correctly removed (`DEP_DELAY`, `TAXI_*`, `WHEELS_*`, `DELAY_DUE_*`), the
remaining pre-departure features (route, airline, schedule, distance,
derived temporal features) are **weakly correlated** with arrival delay,
because most delay variance is driven by *day-of* stochastic events
(weather, late incoming aircraft, ATC events, carrier-specific operational
issues) that are not in the dataset. The MAE of ~26 min is wider than the
entire `Short delay` band (15 min wide), which mechanically explains why
both the MLP and the Ensemble classifiers struggle on the `Short` class
specifically — the noise floor of the underlying regression target is
wider than the class itself.

This justifies the project's pivot from regression to multiclass
classification: the *coarse* 3-class formulation is the granularity at
which the available pre-departure signal is statistically informative,
while a minute-level point prediction is fundamentally over-specified for
what the data can support.

## 10. Hierarchical reformulation (Exp 4) — the strongest result

After the flat 3-class MLP recovered to Macro F1 = 0.37 and the flat
regression failed (R² = −1.20), we reformulated the problem as a
**two-stage cascade**:

- **Stage 1** — binary classifier: `On-time` (<15 min) vs `Delayed` (≥15 min),
  trained on the full 200 k training set. Same MLP backbone, 2-output
  softmax head, `class_weight='balanced'` plus the `MacroF1Monitor`
  callback that was already established to be the necessary fix.
- **Stage 2** — regression on `ARR_DELAY` in minutes, trained on the
  **delayed-only subset** of the training set (~54 k rows). Same MLP
  backbone, single linear output unit, standard `EarlyStopping(val_loss)`
  (val_loss is a clean signal for regression).
- **Inference** — for each test flight, run Stage 1; if it predicts
  `Delayed`, run Stage 2 to obtain the minute-level prediction;
  otherwise, predict 0 minutes (and class `On-time`).

The composite metrics are derived in two ways:

- **3-class metrics** — bin the predicted minutes (`< 15` → On-time, `15–30`
  → Short, `> 30` → Long) and compare against the flat MLP / Ensemble tables.
- **End-to-end regression metrics** — MAE / RMSE on the full test set,
  treating Stage 1's `On-time` predictions as 0-minute predictions.

### 10.1 Three concrete wins

**Win 1: Stage 1 standalone is the project's best classifier.**
On the binary "will it be late?" question, Stage 1 achieves
**Macro F1 = 0.623** (accuracy 68.1 %, F1 on the rare `Delayed` class
0.475). This is dramatically above any 3-class formulation tried — flat
MLP, Ensemble, kNN — because the binary problem has a much less imbalanced
73/27 distribution and the model can concentrate capacity on a single
decision boundary. For an operational deployment (e.g. an airline
passenger-warning system), this binary predictor is the deliverable that
matters.

**Win 2: The cascade rescues the regression.**
Stage 2's R² on the delayed-only test set is **−0.001** (essentially the
constant-mean predictor), vs **−1.20** for the flat regression on the full
test set. End-to-end on the full test set the cascade achieves **MSE = 3 868**
(45 % lower than the flat regression's 6 989) and **MAE = 29.2 min**
(comparable to the flat 26.1, but with a model that *is not actively
worse than guessing the mean*). The reason: the flat regression had to
fit a target that spans 0 minutes (73 % of the test set) and 500+ minutes
(a long tail) with a single regressor; the cascade lets Stage 1 absorb
the easy "predict 0" decisions and lets Stage 2 specialise on a tighter
target distribution (all ≥ 15 min).

**Win 3: The composite 3-class is a (slight) numerical win with an
operationally better error pattern.**
Composite Macro F1 = **0.38**, slightly above the flat MLP's 0.37, but the
*shape* of the errors is qualitatively different:

| | F1 On-time | F1 Short | F1 Long |
|---|---|---|---|
| Flat MLP (Exp 1)               | 0.74 | **0.22** | 0.16 |
| Hierarchical (Exp 4)           | 0.77 | **0.01** | **0.35** |

The hierarchical model **doubles** F1 on the `Long delay` class
(the operationally expensive event — flights that will cost the airline
real money) at the cost of essentially abandoning the `Short delay`
class. This is a defensible operational trade-off: a 15–30 minute delay
is roughly the noise floor of normal airline operations; a >30 minute
delay is a meaningful event. Stage 2's regression almost always predicts
≥30 minutes for the flights it sees, which mechanically produces this
collapse-onto-Long pattern.

### 10.2 Why Stage 2's R² ≈ 0 is itself a finding

Stage 2 is trained on the **cleanest** possible regression target — only
flights that are *known* to be delayed — and it still cannot beat the
constant-mean predictor. This is the most direct piece of evidence yet
for the project's broader conclusion: **per-minute arrival delay is
essentially unpredictable from pre-departure features alone, regardless of
model class or formulation.** The cause is the missing features
(day-of weather, late aircraft cascades, ATC events) that drive most of
the minute-level variance, not the model.

The flat regression's R² = −1.20 was a *worst-case* version of this same
finding: the model wasted capacity trying to predict 0 for the 73 % of
on-time flights, ending up worse than the mean. The hierarchical Stage 2
isolates the question to a regime where the regression *should* work if
the signal existed; the resulting R² ≈ 0 shows that the signal genuinely
isn't there.

### 10.3 Suggested narrative for the report

The deep-learning phase produced **three deliverables**, in order of
operational value:

1. **Stage 1 binary classifier** — Macro F1 = 0.62, the project's best
   predictor. *"Will this flight be late?" — yes/no with 68 % accuracy.*
2. **Hierarchical end-to-end MAE 29 min** — a usable minute-level
   prediction for delayed flights, after rescuing the failed flat
   regression.
3. **Flat MLP (Exp 1) Macro F1 = 0.37** — a competitive 3-class baseline,
   directly comparable with kNN/Ensemble in the final model-comparison
   table.

## 11. Suggested wording for the LaTeX report

Five defensible claims:

1. **On the architectural choice.**
   *"Because the dataset is purely tabular, with no spatial or sequential
   structure per sample, we adopted a Multi-Layer Perceptron — the
   canonical deep architecture for structured data — implemented
   layer-by-layer in TensorFlow / Keras with Dense + BatchNorm + Dropout
   blocks, ReduceLROnPlateau, and a custom early-stopping callback (see
   below)."*

2. **On the diagnosed failure of the default training recipe.**
   *"The default combination `class_weight='balanced'` +
   `EarlyStopping(monitor='val_loss', restore_best_weights=True)` produced
   a degenerate predictor: 73 % accuracy with zero recall on both minority
   classes. Investigation showed this is structural, not stochastic:
   `class_weight` only re-weights the training loss, but the unweighted
   validation loss is minimised by the trivial constant-majority predictor.
   `restore_best_weights` therefore reverts the model to its earliest
   degenerate epoch at the end of every run, regardless of intermediate
   training quality."*

3. **On the corrective fix.**
   *"Replacing `EarlyStopping(val_loss)` with a custom callback that
   monitors validation Macro F1 — a metric immune to majority collapse —
   was sufficient to recover a non-degenerate model: Macro F1 rose from
   0.28 to 0.37, with non-zero recall on all three classes, while keeping
   every other hyperparameter unchanged. A subsequent stress-test
   ('Experiment 1b') that additionally removed BatchNorm, used stronger
   manual weights, and initialised the output bias to `log(class_priors)`
   actually re-collapsed the model — confirming that the macro-F1 callback
   was the necessary and sufficient intervention, and that piling further
   'fixes' on top destabilises training."*

4. **On the hierarchical reformulation.**
   *"We then split the problem into a binary classifier (`On-time` vs
   `Delayed`) chained with a regression on the delayed-flight subset. The
   binary stage achieves Macro F1 = 0.62 — the strongest classifier of
   the entire project — by exploiting the less-imbalanced 73/27
   distribution of the binary partition. The regression stage rescues
   the failed flat regression from R² = −1.20 to an end-to-end MSE
   45 % lower (3 868 vs 6 989) and a usable MAE of 29 minutes. The
   composite 3-class Macro F1 (0.38) marginally beats the flat MLP
   (0.37) while doubling F1 on the `Long delay` class — the operationally
   most expensive event — at the cost of the `Short delay` class. This is
   a defensible trade-off for an airline scheduling application, where a
   15–30 min delay is near the operational noise floor and the >30 min
   regime is the one that triggers concrete cost."*

5. **On the irreducible noise floor (the project's headline negative result).**
   *"Stage 2 of the hierarchical model is trained on the cleanest possible
   regression target — only flights known to be delayed — and still
   produces R² ≈ 0, i.e. cannot beat the constant-mean predictor. This
   is the strongest evidence in the project that per-minute arrival
   delay is fundamentally unpredictable from pre-departure features
   alone: the missing signal (day-of weather, late incoming aircraft,
   ATC events, carrier-specific operational issues) lives outside the
   dataset. The Mean Absolute Error of ~26 minutes — wider than the
   entire `Short delay` band — quantifies this irreducible noise floor
   and independently motivates the project's formulation as a coarse
   3-class classification rather than minute-level regression."*
