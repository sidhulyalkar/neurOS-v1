# Matched repetition behavioral preflight (research v2)

**State:** discovery-only experimental protocol, not an empirical or causal
circuit finding. This deliberately has a **different scientific identity**
from the v1 Pythia pilot. It addresses [Issue #207](https://github.com/sidhulyalkar/neurOS-v1/issues/207).

A model may assign a positive next-token logit margin to D without using
repetition. A raw positive margin does not prove copying or induction.

## Frozen paired conditions

Each example has eight distinct single-token words A through H (E is foil):

| Condition | Seven-token prompt | Control |
| --- | --- | --- |
| Repeated | A B C D A B C | Target repeated order |
| Reordered | A B C D B A C | Same token counts, last C, initial context, scoring target and foil |
| Novel | A B C D F G H | New suffix, secondary sensitivity control |

Every condition scores the same next-token D versus E logit margin. The
reordered case is the primary lexical control; the novel case also changes
the token inventory and last context token, so it is confounded.

**Identifiability limit:** both the repeated and reordered conditions end in
C, and both have an earlier C followed by D. A legitimate single-token
induction mechanism that matches C and copies the following D may predict
D in **both** conditions and fail this ordered-repeat gate. The test measures
higher-order order-sensitive repetition above that simpler copying behavior.
A rejection does **not** rule out induction; a passing gate does not establish
that any particular attention head implements it. The separate novel-suffix
comparison partly tests single-token repetition but changes several
confounders. The offline suite explicitly includes this false-negative case.

The generator freezes eight discovery and eight validation trials from
disjoint lexical pools. **The runner never scores validation model outputs.**
Its frozen input IDs are hashed for later independent reproduction.

## Predeclared discovery gate

For each example, record raw repeated, reordered and novel margins.
Compute the paired advantages repeated minus reordered and repeated minus
novel. The gate requires a positive repeated margin on at least 75% of
examples, positive paired advantages against both controls on at least
75%, and positive mean advantage against both controls. Paired bootstrap
intervals resample examples, not measurements as independent trials.

A positive raw token prior without repetition **must fail**. A planted
copy-sensitive synthetic predictor can pass. Neither establishes a true
Transformer circuit. There are only eight discovery trials, so no
confirmatory statistical or mechanistic claims are licensed.

## Run on a pinned public checkpoint (explicit opt-in)

~~~bash
pip install -e "packages/neuros-mechint[real-model-pilot]"
python packages/neuros-mechint/examples/25_pythia_matched_repetition_preflight.py \
  --revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c \
  --device cpu \
  --out ./pythia-matched-repetition-discovery-v2
~~~

This does **not** execute as part of normal CI. The preflight writes
matched-inputs.json and status.json, preserving rejected outcomes. The public
revision must be accessible and independently rechecked at execution time.
The pretrained weights have not been executed in this work.

## What still needs building

After a frozen discovery candidate, evaluate held-out intervention
difference-in-differences:

~~~text
(repeat - reordered)_intact - (repeat - reordered)_patched
~~~

Use the same intervention target universe across conditions, include
same-cardinality random controls, discovery-only candidate selection,
zero/mean and matched-distribution donors, v2 negative evidence artifacts
and independent reproduction. A passing discovery behavior gate is not
permission to inspect held-out outputs before candidate freeze.

The current seven-token synthetic task cannot support claims about
biological homology, general induction circuits or clinical BCI deployment.
