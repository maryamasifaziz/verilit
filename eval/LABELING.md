# Labeling guide

Each row of your sheet is one claim and the abstract of the paper it cites. Decide whether
**the abstract shown** supports the claim. Judge only from that abstract: not from what you
know, not from the paper's full text, and not from what seems likely.

Write one letter in the `label` column:

| Label | Use when | Example |
|---|---|---|
| **S** (Supported) | The abstract states everything the claim says. Paraphrase is fine. Every number, dataset, model and direction of effect matches. | Abstract: "...reduces hallucination by 21% on TruthfulQA". Claim: "X reduces hallucination on TruthfulQA by 21%." |
| **C** (Contradicted) | The abstract states something incompatible with the claim: a different number, the opposite direction, a different dataset or method for the same result. | Abstract says 85% accuracy; claim says 95%. |
| **I** (Insufficient evidence) | The abstract doesn't state the claim, or states only part of it, or the claim adds a detail (number, comparison, generalization) the abstract doesn't give. | Claim says "outperforms GPT-4"; the abstract never mentions GPT-4. |

Rules of thumb:

- **Partly supported counts as I.** If any part of the claim is missing from the abstract, it's I.
- **Overgeneralizing counts as I.** "Method X reduces hallucination" when the abstract only reports
  it on one benchmark is fine; "Method X eliminates hallucination" when it reports a reduction is I.
- **Rounded or changed numbers count as C** if they're clearly different, I if the abstract gives no number.
- **Hedged vs. certain:** if the abstract says "may improve" and the claim says "improves", use I.
- If you can't decide, choose your best guess and write why in `notes`.

Work alone, and don't look at the VeriLit reports or at each other's sheet until you're both
done; the 30 rows that appear in both sheets measure how much you agree, and that only works
if you label them independently. The sheets don't show the Verifier's verdict on purpose.

Rows are shuffled across questions. Save the file as CSV (UTF-8) when you're done; keep the
`item_id` column unchanged.
