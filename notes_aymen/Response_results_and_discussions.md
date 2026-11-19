# Response to `results_and_discussions.md`

Reply to Aymen's notes on Chapters 4 (Results) and 5 (Discussion). Explanations are short; sentences in *italics* or in quote blocks are drafts in English that can go into the report after rewording. All numbers come from `Synthesis_recent_papers.md` (checked against the PDFs). Items marked **to verify** could not be checked in the papers I have.

Decision taken: keep all studies except Gardezi, no A/B/C grades, and one short paragraph of factual limitations (§4).

---

## 1. The "apples to oranges" problem: turn it into a finding

You cannot say "model X is best on task Y with dataset Z", because almost no two papers share the same dataset, task and metric. That is normal for a review of this field, and it is not your fault. Three ways to handle it:

**a) It answers your own RQ2.** RQ2 asks *"what gaps remain in current evaluation protocols?"*. The fact that results cannot be compared **is** the main answer. Say it explicitly instead of hiding it:

> *Among the 10 studies, no two share the same dataset, task and metric. Results can therefore only be compared within each study, against the baselines its authors chose.*

(Check: MEGA and Sumsuzzaman both use IBM AML HI-Small but report different metrics, minority F1 vs AUPRC, so the sentence holds.)

**b) Compare inside each paper, not across papers.** For each study, ask only: *did the proposed GNN beat its own strongest baseline, under the same data and split?* The answer is "better / about equal / worse". This is called *vote counting by direction of effect*; it is the standard fallback in narrative reviews when effect sizes are not comparable. It needs only the paper's own results table, which you already read.

| Study | Strongest non-GNN baseline in the paper | Same features and split? | GNN vs that baseline |
|---|---|---|---|
| Silva | LightGBM / XGBoost | Yes | Worse (AP 0.783 vs 0.967) |
| Yue | Gradient boosting on graph features | Yes | About equal (AP 0.262 vs 0.262; F1 0.265 vs 0.271) |
| Dintakurthy | XGBoost | Yes | Worse alone (PR-AUC 0.075 vs 0.425); better when combined (0.450) |
| Sumsuzzaman | MLP, XGBoost | Yes | Worse (AUPRC 0.166 vs 0.210 MLP, 0.172 XGBoost) |
| Nguyen & Le | LightGBM | **No**: the GNN gets extra entity features | Better (AP 0.641 vs 0.361) |
| Chen & Yang | MLP with engineered features | Only an MLP, no tree model | Better (P@20%R 86.2 vs 68.5) |
| Bellei | Sub2Vec (random-walk embeddings) | Yes | Better (PR-AUC 0.208 vs 0.022) |
| LAS-GNN, MEGA | Only other GNNs | — | Not tested against tabular models |

**c) Map the evidence instead of ranking models.** Place each study by *what it outputs* and *what data it uses*. This is objective, readable from abstracts, and shows the gaps at a glance:

| Output level | Real, public data | Private data | Synthetic data |
|---|---|---|---|
| Group (subgraph) | Bellei (Elliptic2) | — | — |
| Pattern member (key account) | — | — | LAS-GNN |
| Transaction / edge | MEGA (ETH, node level) | — | MEGA, Silva, Sumsuzzaman (IBM AML / AMLSim) |
| Transaction / node | Yue (Elliptic), Nguyen & Le (IEEE-CIS) | Chen & Yang | Nguyen & Le (PaySim), Dintakurthy (BankSim) |

Reading: group-level detection on real data = **one study**, on Bitcoin. Nothing at group level on bank data.

**Proposed split between chapters**
- **Results** = what the studies report: the results table, the evidence map, the direction-of-effect counts, which metrics are used. No judgement.
- **Discussion** = what it means: which gaps from the reviews are closing, why GNNs do not beat trees, limits of the evidence, open problems.

---

## 2. Answers to your questions, point by point

### Deployment (latency, scale)
Not worth much space. At most one sentence in the Discussion, or drop it:
> *Several studies now report inference latency or throughput, a sign that the field is moving toward deployment, but these measurements use different hardware and cannot be compared.*

### Typology table (§2.7) and "strongest evidence"
"Strongest evidence" meant *the most convincing study for this typology*. Agreed: replace the table with a few sentences. Careful with your draft "LAS-GNN and MEGA have both been applied to rings and structuring": it is true for LAS-GNN, not quite for MEGA. MEGA is tested on IBM AML, whose laundering includes cycles and scatter-gather patterns, but it reports one global score, not a score per pattern. Suggested version:

> *Only one recent model explicitly targets ring and structuring patterns: LAS-GNN detects the key accounts of cycles and scatter-gather schemes, but only in synthetic networks @verlaan_-gnn_2025. Multigraph models such as MEGA are evaluated on synthetic data that contain these patterns, but report a single global score @bilgi_mega_2026. Complex multi-node laundering on real data is studied only on Bitcoin, through the subgraphs of Elliptic2 @bellei_shape_2024.*

### Temporal modeling: "many techniques, need to synthesize"
Group them by *how time is used*, not by technique name:
> *Time now enters GNN fraud models in three ways: as the order of transactions inside a pattern @verlaan_-gnn_2025, as timestamps encoded in attention or recurrent layers @chen_real-time_2026 @nguyen_real-time_2025, and as a data stream on which the model is updated when its performance drops @yue_adaptive_2026. Most recent studies also split data chronologically, so that models are never trained on future transactions.*

### Heterogeneous graphs
Agreed, keep it short. Your sentence "No study uses many entity types on a real banking graph with public results" is enough, with one preceding clause such as *"A few studies add entity types such as cards, merchants or devices to the transaction graph @nguyen_real-time_2025 @dintakurthy_multisource_2026."*

### Metrics: your draft sentence
Two corrections: "Area Under Curve" is ambiguous (there is ROC-AUC and PR-AUC), and ROC-AUC is precisely the metric that stays misleading under imbalance. AP and PR-AUC are almost the same thing. Suggested version:

> *The existing reviews pointed out that fraud datasets are highly imbalanced. Recent studies no longer report accuracy; they use the area under the precision–recall curve (PR-AUC, or average precision), the F1-score of the fraud class, and the share of frauds found among the top-ranked alerts (precision or recall at k). Some also define their own operational metrics. This variety, combined with different datasets and tasks, prevents direct comparison across studies.*

A nice example if you want one: in Dintakurthy's study, logistic regression has the *highest* ROC-AUC (0.856) but a PR-AUC of only 0.265, far below XGBoost (0.425). ROC-AUC can look good while the model finds few frauds.

### GNN vs tabular models: move to Discussion, neutral wording in Results
Agreed. In **Results**, only facts:
> *Four studies compared GNNs with gradient-boosted trees trained on the same features and split chronologically. In none of them did a GNN alone obtain a better score (@tab:results): … Two studies report large GNN gains, but compare them with a model that received less information @nguyen_real-time_2025 or only with a multilayer perceptron @chen_real-time_2026.*

In **Discussion**, the interpretation (see §3).

### "Weak baselines", simplified
Ignore the details. The whole idea fits in one line: *the studies that report large GNN gains did not compare their GNN with a tree model trained on the same information.* Nguyen & Le give the GNN extra customer features (including how many past frauds a customer had) that LightGBM does not get; Chen & Yang compare only with an MLP.

### References for "GNN vs gradient boosting is unresolved"
- Inside the corpus: `silva_efficient_2026`, `yue_adaptive_2026`, `dintakurthy_multisource_2026`, `sumsuzzaman_graph_2026`.
- Earlier: Weber et al. 2019 (`weber_anti-money_2019`), the original Elliptic paper, where random forests outperformed the GCN (**to verify** in their results table).
- The opposite claim: Cheng et al. (`cheng_graph_2025`) write that GNNs "significantly outperform" traditional methods. Motie & Raahemi are more cautious (traditional ML can stay competitive on small or weakly structured graphs).
- Optional, general machine learning: Grinsztajn et al. 2022, *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS 2022). Not in the bib yet.

### "Group output is mostly post-hoc"
Plain explanation: the model gives a risk score to **each transaction**. Afterwards, an algorithm (or an analyst) links the flagged transactions together to form groups, called cases. The group itself is never evaluated by the model; it is a by-product. The cost: groups contain many innocent transactions. In Silva's study, when the top 1% of transactions are inspected, only 7.6% of the transactions in the assembled cases are illicit, and the share falls to 0.4% at 10%. The only exception in the corpus is GLASS on Elliptic2, which gives a score to the group directly.

> *In banking settings, groups are not detected by the model: they are rebuilt after scoring, by linking high-risk transactions. These rebuilt groups quickly fill up with legitimate activity @silva_efficient_2026. Only on Bitcoin data has a model been trained to score a group as a whole @bellei_shape_2024.*

### "Structuring has no real labels"
Structuring means splitting a large amount into many smaller transfers that stay below reporting thresholds; when the transfers go through several accounts or people, it is often called *smurfing*. In the corpus:
- it appears as **synthetic patterns** (fan-out, scatter-gather in IBM AML; scatter-gather in LAS-GNN's generated graphs);
- it appears as an **observation**, not a label, in Elliptic2: the authors found "peeling chains" (the Bitcoin analogue of smurfing) in subgraphs the model flagged;
- no dataset labels a real structuring case.

A useful nuance for the Discussion: structuring done by one account (many amounts just below a threshold) is visible in a simple table of transactions; a graph only becomes necessary when the split runs over many accounts.

> *No public dataset contains real cases of structuring labeled as such. Recent evidence is either synthetic @verlaan_-gnn_2025 or qualitative, such as the peeling chains observed in suspicious Bitcoin subgraphs @bellei_shape_2024.*

### "Pattern-aware GNNs tested where patterns define labels"
Fine to drop. The idea is already in your Datasets section (main.typ, end of the IBM AML paragraph: "its labels are generated by the same patterns that these models are designed to find"). That sentence is enough.

### Unsupervised / label-free detection
Yes, it means learning **without fraud labels**. All 10 studies are supervised: they all need labeled frauds for training. Motie & Raahemi already called for unsupervised methods. Among the papers in your bib, Audit-HCL (`jiao_dynamic_2026`) claims it still works with zero fraud labels (abstract only, not read in full). Good open problem for the Discussion, because real laundering labels arrive late and are rare.

### Reproducibility
- **Code**: public for 2 of 10 studies (Bellei, MEGA). Yue describes an "artifact package" but gives no link; Dintakurthy shares data "upon reasonable request"; the others say nothing.
- **Data**: 9 of 10 use public or synthetic data that others can obtain; Chen & Yang use private industrial data.
- **Comparison with the reviews**: Motie & Raahemi found that about 15% of their papers released code. 2 of 10 is about the same: no visible progress.

> *Only two of the ten studies release their code @bellei_shape_2024 @bilgi_mega_2026, a proportion close to the one observed by Motie and Raahemi @motie_financial_2024. Reproducing or comparing recent results therefore remains difficult.*

### Bank secrecy
Yes, worth a few sentences, but frame it as **legal and privacy constraints** rather than "culture", because the reasons are legitimate: data protection law, banking secrecy, and competition. Since you are at HES-SO, a Swiss example is natural (banking secrecy in the Swiss Banking Act, art. 47: **to verify** and cite). Consequences already visible in your corpus: synthetic generators replace real data (the IBM AML generator was built for this reason, `altman_realistic_2023`, **to verify** in their introduction), and the only bank-scale result comes from private data that nobody can check (Chen & Yang). Possible ways forward to mention: realistic synthetic data, federated learning (training across banks without sharing data), shared benchmarks.

### Li et al.
Agreed: keep the existing citations, nothing more.

### Small fixes in your notes file
Line 1 links to `Synthesis_existing_reviews.md`; it should be `Synthesis_recent_papers.md`.

---

## 3. Discussion: the GNN vs tabular point, interpreted

A short interpretation you can adapt:

> *These results do not show that graphs are useless. They suggest that, on transaction data with rich features, a well-tuned tree model already captures most of the signal, and the graph adds little when used alone. Graphs help most in two situations: when fraud is defined by a structure (a cycle, a laundering path, a group), and when their output is combined with a tabular model @dintakurthy_multisource_2026. A fair evaluation of GNNs for fraud therefore needs a gradient-boosting baseline trained on the same features and the same chronological split.*

The last sentence can become a recommendation in your conclusion (a minimum standard for future studies: public dataset, chronological split, tree baseline, PR-AUC).

---

## 4. Limitations of the included studies (replaces grades and the long error list)

Only facts that can be read in the abstracts, tables or data descriptions; no judgement on venues or authors. Suggested location: end of the Methodology chapter (after the AI-use section) or start of the Discussion.

> *The evidence from recent studies has clear limits. Five of the ten studies rely mainly on synthetic data, and one on private industrial data that cannot be checked. Some test sets are very small (16 frauds in one study @dintakurthy_multisource_2026), and only two studies release their code. In addition, headline numbers must be read with care: the F1-score reported for Elliptic2 is a micro-average, which in practice equals accuracy, while the precision–recall measure of the same model is much lower @bellei_shape_2024; and in one study the GNN uses customer features derived from past fraud labels that its baseline does not receive @nguyen_real-time_2025. We report all results as published, and we draw conclusions only from patterns that hold across several studies.*

These are the three most important issues; the rest of the error list in `Synthesis_recent_papers.md` §6 can stay in the notes.

---

## 5. Feedback on the current Chapter 4 text in main.typ

- Typos: "emrged" → "emerged"; "taht" → "that"; double spaces in "Elliptic2  dataset" and "MEGA,  preserves".
- "the Elliptic2 dataset shows that…" → "experiments on Elliptic2 show that…" (a dataset does not show anything; the experiments do).
- "Some of the review papers claimed that GNN consistently outperformed…" → only Cheng et al. say this; name and cite them: "Cheng et al. @cheng_graph_2025 state that GNNs outperform…".
- "GNN" → "GNNs" (plural) in that sentence.
- The GNN vs tabular paragraph contains interpretation ("Large GNN gains are reported mainly against weak or unequal baselines"); per your own rule, move this sentence to the Discussion and keep the factual version from §2 in Results.

---

## 6. Proposed outlines

**Chapter 4: Results**
1. Short intro + `tab:results` (done).
2. Output level of the studies: the evidence map (§1c) as a small table or two sentences.
3. Comparison with baselines: direction-of-effect counts (§1b), including the four tree-model comparisons, stated neutrally.
4. Metrics and protocols: which metrics are used, chronological splits, and the fact that no two studies share dataset + task + metric.

**Chapter 5: Discussion**
1. Which gaps from the reviews are closing: subgraph level (partly, crypto and synthetic only), temporal (yes, three uses of time), heterogeneous (little), metrics (better, but varied).
2. GNNs vs tree models: interpretation (§3).
3. Limitations of the evidence: factual paragraph (§4) + reproducibility.
4. Open problems: no real bank data with group labels (privacy and secrecy), no real structuring labels, groups rebuilt after scoring, no unsupervised group detection.
5. (Optional) recommendation: minimum evaluation standard for future studies.
