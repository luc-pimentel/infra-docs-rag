# Postmortem: the judge read the passages without their headings

**What:** the grounding judge scored two correct, fully cited answers as fabricated (0.86 and 0.53
probability of an added fact), and the benchmark counted them against the pipeline for one run.
**Where:** stage 07's judge prompt, found in stage 08's calibration.
**Impact:** two of 54 judged answers misread; no user saw it (the judge is a reading, not a verdict,
and changes no answer). Had the regression gate on `grounding.judge_fabricated` been live, a 2-of-54
change against a baseline of 1 would have blocked a release that had done nothing wrong.
**Fixed in:** stage 08, three revised rules, one of them this.

## Timeline

- **Stage 07.** The judge (Jev, a model that returns calibrated probabilities and never text) is given
  the question, the answer, and each passage's text. The passages reach the generator under a numbered
  heading (`[2] Kubernetes › Deployments › Rolling Back a Deployment, pages 9–10`), because the heading
  is what an answer about "page 9" or "the Deployments section" rests on. The judge's state builder
  passed the passage text and left the heading out: the code that assembled the generator's prompt and
  the code that assembled the judge's were written separately, and the judge's was shorter.
- **Stage 07 run.** 54 answers judged. The two page questions (`whitepaper-page`, "What is on page 9
  of the Cloud Native Security Whitepaper?", and `supply-chain-page`) came back with fabricated
  probabilities of 0.53 and 0.86. Both answers cited the right passages and said, correctly, what was
  on the page. The report flagged them as "the judge and the mechanical checks disagree" and moved on,
  since the judge changed no verdict.
- **Stage 08, calibration.** Twenty-two answers labelled by hand on five axes, chosen to include every
  disagreement. Both page answers labelled grounded, not fabricated. Reading the judge's input for them
  made the cause obvious: the answer said "page 9 covers the cloud native layers", and the judge saw a
  passage about cloud native layers with no page number anywhere in it. From its point of view, "page
  9" was a fact from nowhere. That is a fabrication by its definition, and a correct reading of the
  wrong input.
- **Fix.** The judge's state builder was changed to write each passage under the heading the generator
  saw it under. Re-judging the two answers with the headings in place: `supply-chain-page` moved from
  grounded 0.16 / fabricated 0.86 to 0.59 / 0.20; `whitepaper-page` from 0.59 / 0.53 to 0.79 / 0.19.
  Paraphrase questions, whose answers do not depend on the heading, did not move (`prom-naming`,
  grounded 0.47 before and 0.48 after). The saved readings in the stage 07 run were left as they were
  and the report says so; the stage 08 runs were made with the fix.

## Why it was not caught earlier

- The judge was added as an extra signal that changed no verdict, so its disagreements were reported
  rather than investigated. "Reported rather than investigated" is the state everything is in until a
  person reads it; calibration against hand labels is the step that makes someone read it.
- The two prompts were built by two functions. The generator's had been iterated on through stage 07;
  the judge's was written once. Nothing checked they agreed on what a passage is.
- The symptom only shows on questions whose answer is *about* the passage's metadata rather than its
  text: two of 54. A judge that is wrong on 4 % of the set in a way that looks like it is doing its job
  is not a judge anyone doubts.

## What changed beyond the fix

- One function builds a passage's representation for both the generator and the judge
  (`Source.heading`, used by `prompt.passages` and `judge.state`); a test asserts the judge's input
  contains each heading.
- The calibration set is a committed file and `calibrate` is a command: the next judge change gets
  measured against the same 22 labels, and the report lists every disagreement with the answer in full.
- The judge's grounded floor was recalibrated in the same pass (0.70 → 0.50), since nothing under 0.70
  was ungrounded by hand. Same cause family: a threshold chosen before anyone read the cases.
- The next calibration should seed negatives: every answer in this set was grounded by hand, so the
  set measures the judge's false alarms and nothing about its recall. That is still open.

## What would have caught it in production

The trace (stage 10) carries the judge's input and reading next to the answer, and `traces --summary`
counts readings in the unsure band. A page question answered right and judged fabricated would be one
flagged trace with the heading visibly missing from the judge's view. The gate on
`grounding.judge_fabricated` (stage 08) would have refused a release over it, which is the right
outcome for the wrong reason: the gate cannot tell a judge regression from a pipeline regression, and
only the calibration can. That is why the judge is a signal in the gates and not the verdict.
