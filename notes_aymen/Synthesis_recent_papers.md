# Synthesis of recent papers (2024–2026) for Chapter 4 "Results"

Purpose: show what the 2024–2026 literature adds on top of the three existing reviews (Motie & Raahemi 2024, Cheng et al. 2025, Deprez et al. 2025; see `Synthesis_existing_reviews.md`), and answer the research question of the title: *which GNN architectures detect illegal financial rings, structuring, and complex multi-node fraud in transaction networks?*

Scope: the 11 papers below, all read in full. Li et al. (2025) is a review, not a primary study; it is moved to the reviews section (§8). Citation keys match `report/refs.bib`.

| Key | Paper | Venue |
|---|---|---|
| `bellei_shape_2024` | Bellei et al., *The Shape of Money Laundering* (Elliptic2) | KDD MLF '24 workshop (arXiv 2404.19109) |
| `verlaan_-gnn_2025` | Verlaan, Hulpuș, van Leeuwen, *LAS-GNN* | ACM ICAIF '25 |
| `bilgi_mega_2026` | Bilgi & Atasu, *MEGA* | TMLR (08/2026) |
| `silva_efficient_2026` | Silva & Lima de Castro, *Graph-Based Prioritization* | ICEIS 2026 |
| `nguyen_real-time_2025` | Nguyen & Le, *Heterogeneous Temporal GNN* | ICAART 2025 |
| `yue_adaptive_2026` | Yue et al., *ATGNN-SP under Concept Drift* | Highlights in Business, Economics and Management, vol. 69 (2026) |
| `chen_real-time_2026` | Chen & Yang, *Real-time dynamic graph learning (C2GAT)* | Frontiers in AI 9 (2026) |
| `dintakurthy_multisource_2026` | Dintakurthy, *Multisource Fraud Intelligence* | J. Computing and Data Technology (Nov. **2025**) |
| `sumsuzzaman_graph_2026` | Sumsuzzaman et al., *GNN Fraud and Cybersecurity in Banking* | Int. J. Adv. Signal and Image Sciences (2026) |
| `gardezi_large-scale_2025` | Gardezi & Sawant, *LS-GAF* | ICCCSS 2025 |
| `li_graph_2025` | Li et al., *Graph Learning-Empowered FFD* (review) | Intelligent Computing 4 (2025) |

---

## 1. Corpus overview (10 primary studies)

| Paper | Detection level (output) | Typology targeted | Graph | Data | Split | Tabular baseline? | Main metric | Code / data | Grade |
|---|---|---|---|---|---|---|---|---|---|
| Bellei 2024 | **Subgraph** (binary class of a labeled subgraph) | Multi-node laundering paths; peeling chains (≈ smurfing) | Bitcoin cluster graph, 49M nodes / 196M edges, 122K labeled subgraphs | Real (Elliptic2) | Random 80/10/10 | No | PR-AUC, ROC-AUC, micro-F1 | Public data + code | A |
| Verlaan 2025 (LAS-GNN) | **Motif key node** | Rings (directed cycles C3–C6), scatter-gather (smurfing), gather-scatter | Directed, weighted, timestamped simple graph | Synthetic (Watts–Strogatz, LFR) | Separate train/val/test graphs | No (GNN baselines only) | AP, F1 | Not stated | A |
| Bilgi 2026 (MEGA) | **Edge** (AML), node (ETH) | IBM AML patterns (incl. cycle, fan-in/out, scatter-gather) as edge labels | Directed multigraph with edge attributes | IBM AML (synthetic, 6 sets) + ETH phishing (real) | IBM standard split; ETH temporal 65/15/20 | No | Minority-class F1 | Public code | A |
| Silva 2026 | **Edge** ranking + post-hoc **case** (WCC) | Multi-node cases | Directed multigraph | Synthetic (AMLSim 100k / 1M accounts) | Out-of-time 60/20/20 | **Yes** (MLP, XGBoost, LightGBM) | Recall@k, IPI@k, case purity/coverage | Not stated | B |
| Nguyen & Le 2025 | Transaction node | None specific (card fraud) | Heterogeneous temporal graph (transaction + entity nodes) | IEEE-CIS (real), PaySim (synthetic) | Sliding temporal windows | Yes (LightGBM) | AP, ROC-AUC, latency | Not stated | B |
| Yue 2026 | Transaction node | None specific | Directed snapshots (Elliptic, structure only) | Real (Elliptic) | Chronological 1–29 / 30–34 / 35–49, prequential | **Yes** (LR, RF, HGB, graph HGB) | AP, F1 | Artifact claimed | B |
| Chen & Yang 2026 | Transaction (buyer–seller) | Fund cycles, mule chains (cash-out) via path subgraph | Continuous-time dynamic graph | Industrial (Ant: Taobao + offline merchants), public link-prediction sets | Months 1–3 train / 4–6 test | MLP + handcrafted features only | P@20%R | No | C |
| Dintakurthy 2025 | Transaction edge + entity max-aggregation | None specific | Account–merchant bipartite | Synthetic (10,000-row BankSim extract) | Chronological by step | **Yes** (LR, XGBoost) | PR-AUC, F1, MCC | Claimed | C |
| Sumsuzzaman 2026 | Edge + node risk tiers | None specific | Directed multigraph | IBM AML HI-Small (unnamed, synthetic) + NetFlow | Temporal 70/15/15 (AML) | **Yes** (LR, RF, XGBoost, MLP) | AUPRC, P@k | Not stated | C |
| Gardezi 2025 | Transaction | Claims "collusion groups" in XAI layer | Heterogeneous temporal, federated | Five unnamed datasets | Not described | No | Precision, recall, F1, AUC | No | C |

Observations:
- Only **one** study produces a group as the model output (Bellei, subgraph classification). One produces pattern membership at node level (LAS-GNN). The others score nodes or edges; groups, when they appear, are rebuilt afterwards (Silva's connected components, Sumsuzzaman's dense high-risk components).
- **5 of 10** studies rely mainly on synthetic data (IBM AML, AMLSim, BankSim, PaySim, Watts–Strogatz/LFR). Real data: Elliptic, Elliptic2, IEEE-CIS, ETH, and non-public industrial data.
- **4 of 10** compare against a strong tabular model (gradient boosting) under a chronological split: Silva, Yue, Dintakurthy, Sumsuzzaman. All four find that the GNN does not clearly win (§3).

---

## 2. Advances against each gap identified by the reviews

### 2.1 Gap: no subgraph / community-level detection (M&R; confirmed by Cheng et al. and Deprez et al.) → partly closed

**Dataset.** Elliptic2 (Bellei 2024) is the first large labeled benchmark for subgraph classification in finance: 121,810 labeled subgraphs (2,763 suspicious, ≈2%) inside a background graph of 49.3M Bitcoin clusters and 196.2M edges, i.e. almost three orders of magnitude larger than earlier subgraph datasets. The authors argue that AML is "inherently" a subgraph problem and that node-level methods work at the wrong level of abstraction. This is a direct answer to M&R's remark that no reviewed paper covered abnormal communities.

**Architecture.** On Elliptic2, a GNN trained on subgraphs as isolated graphs (GNN-Seg) is near random (test PR-AUC 0.026, ROC-AUC 0.537), while GLASS, which adds a 0/1 "labeling trick" feature marking subgraph membership and uses the background graph, reaches PR-AUC 0.208 and ROC-AUC 0.889. Two lessons: (i) plain pooling of node embeddings is not enough to detect a group; (ii) context outside the subgraph matters. External validation: an exchange checked 52 accounts flagged by the model; 14 (26.9%) were linked to illicit activity versus <0.1% of its customers overall. The paper also reports peeling chains (the crypto analogue of smurfing) and nested services in suspicious subgraphs, and uses predictions to trace the funding source (mixers, darknet market, Ponzi scheme).

**Motifs.** LAS-GNN (Verlaan 2025) extends Egressy et al. (ego IDs + direction-aware message passing) to *temporal* motifs: a motif occurrence must respect the chronological flow of funds. Two new components: *signed message passing* (one aggregator; incoming and outgoing messages weighted by learned vectors for +1/−1, optionally scaled by the amount) and an *LSTM aggregator* fed with neighbors sorted by timestamp. On synthetic graphs it detects key nodes of temporal gather-scatter, scatter-gather and 3-cycles with AP 0.97–1.00 (vs 0.86–0.88 for Egressy's GIN+Ego+HMP), and keeps F1 0.87 on 6-cycles where the baseline falls to 0.49.

**Cases.** Silva (2026) turns the top-k% ranked transactions into investigation cases (weakly connected components + induced context). It introduces case-level metrics (coverage, purity, intra-case completeness) and shows a clear trade-off: at k = 1%, coverage 0.88 but purity 0.076; at k = 10%, coverage 0.98, purity 0.004 and average case size above 6,000 edges (AML1M).

**Paths.** Chen & Yang (2026) add a buyer→seller path subgraph (BFS, paths up to length 3) to capture cycles and mule chains; removing it lowers P@20%R from 86.2 to 79.6 on Taobao data (but see §6 on credibility).

**Verdict.** The gap is closed only for crypto (one benchmark, one validated subgraph model) and for synthetic motifs. In banking data, collective detection is still a post-processing step on top of edge scores.

### 2.2 Gap: patterns used only as edge labels (Deprez et al.) → better architectures, same output level

Deprez et al. classify each IBM AML transaction with its pattern label and test only a GCN. The recent line Egressy et al. (Multi-GNN: reverse message passing, port numbers, ego IDs) → MEGA (Bilgi 2026) builds architectures that can actually represent these patterns:
- **Neighbor-aware aggregation**: multi-edges from the same neighbor are aggregated first, then neighbors are aggregated. It is proven strictly more expressive than single-stage aggregation for moment-based aggregators, permutation-equivariant (Multi-GNN's port numbers are not), and has the same asymptotic cost.
- Results: on the 6 IBM AML sets, MEGA beats the base GNN in 24/24 cases and Multi-GNN in 17/18, with best minority F1 78.26% (Medium-HI) and largest gains on the Low-Illicit sets. Ablations show the gain needs the true (source, destination) grouping (random grouping: F1 drops to 9.66% on Medium-LI for MEGA-GIN) and disappears when multi-edges are collapsed on ETH.
- Relevance for **structuring**: repeated transfers between the same pair of accounts are exactly what neighbor-aware aggregation preserves (per-neighbor sums and moments), whereas standard aggregation mixes them with other senders.

Output is still edge-level: the model flags laundering transactions, not rings.

### 2.3 Gap: static graphs dominate (M&R: 50% static) → temporal modeling is now the default

- Continuous-time attention with node-specific Fourier time encoding (Chen & Yang).
- Heterogeneous temporal graph: GAT per snapshot → LSTM over snapshots → attention fusion of entity embeddings; Lambda architecture with an offline BatchNet and an online SpeedNet reading cached embeddings (Nguyen & Le).
- Chronological order inside the motif (LAS-GNN): the temporal order is part of the pattern definition, not a feature.
- Drift-gated stable/plastic model (Yue): a frozen MLP keeps the initial concept; a plastic copy is retrained only when delayed prequential log loss exceeds a limit fixed on validation; replay of an anchor window limits forgetting. This is a lightweight, practical counterpart to Deprez et al.'s continual-learning study. Gains are small (+5.4% relative AP, 3.3 updates over 15 snapshots).
- **Protocol shift**: chronological splits and leakage control are now presented as contributions (Yue: predict-then-update; Silva: message passing restricted to past edges; Dintakurthy: features built only from prior rows). Delayed labels are modeled explicitly (Yue; Chen & Yang report 14–30 days).

### 2.4 Gap: heterogeneous graphs under-studied (M&R: ~30%) → modest progress

Transaction–entity graphs (Nguyen & Le: transaction, card, merchant, device), account–merchant graphs (Dintakurthy), relation types combined with multigraph aggregation (MEGA + R-GCNE), buyer/seller roles (Chen & Yang). No study uses many entity types on a real banking graph with public results.

### 2.5 Gap: imbalance and inappropriate metrics → evaluation is becoming operational

Accuracy has disappeared. Metrics now used: AP / PR-AUC, minority-class F1, P@k, Recall@k, inspections per illicit transaction (Silva: 7.94 at k = 1%), case purity and coverage (Silva), precision at fixed recall (Chen & Yang). Silva shows why: from k = 1% to 10%, recall rises only 0.877 → 0.924 while lift falls 87.7 → 9.2. Remaining issue: some papers still headline micro-F1 (Bellei) or report thresholded F1 = 0 (Sumsuzzaman).

### 2.6 Deployment (latency, scale) → now measured

Chen & Yang: average latency < 30 ms, P99 < 50 ms up to ~20k QPS on a graph of ~900M nodes / 3B edges (unverifiable). Nguyen & Le: 1.8 ms per prediction (LightGBM 0.69 ms). Silva: 24.3M test edges scored in 32.8 s (0.74M edges/s). MEGA: throughput tables; bidirectional message passing costs ~4× throughput. Bellei: training on CPU (1.2 TB RAM, days of training, no features), and a proposal to adapt SALIENT/SALIENT++ to subgraph workloads.

### 2.7 Mapping to the three typologies of the title

| Typology | Pattern in the data | Papers | Strongest evidence |
|---|---|---|---|
| Illegal rings | Directed cycles; funds returning to the originator | LAS-GNN (temporal C3–C6), MEGA (IBM AML cycle pattern), Chen & Yang (path subgraph) | Synthetic only (LAS-GNN, IBM AML) |
| Structuring / smurfing | Scatter-gather; peeling chains; repeated small transfers between the same pair | LAS-GNN (scatter-gather, F1 0.934 weighted temporal), Bellei (peeling chains in suspicious subgraphs), MEGA (per-neighbor statistics) | Real data only as qualitative observation (Bellei) |
| Complex multi-node fraud | Laundering paths, mule chains, cases | Bellei (Elliptic2), Silva (cases), Chen & Yang (mule chains) | Bellei (real data, external validation) |

No paper detects structuring with a real-bank label. No paper outputs a ring as an object on real data.

---

## 3. New finding not reported by the reviews: GNNs do not consistently beat strong tabular models

| Paper | Data | Best GNN | Best non-GNN | Result |
|---|---|---|---|---|
| Silva 2026 | AMLSim 100k | GraphSAGE AP 0.783, R@1% 0.877 | LightGBM AP 0.967, R@1% 0.990 | Trees win clearly |
| Yue 2026 | Elliptic (structure only) | ATGNN-SP AP 0.2624, F1 0.2649 | Graph-feature HGB AP 0.2622, F1 0.2714 | Tie / tree slightly better |
| Dintakurthy 2025 | BankSim 10k | GraphSAGE alone PR-AUC 0.075 | XGBoost PR-AUC 0.425 | Trees win; fusion XGB+GNN 0.450 best |
| Sumsuzzaman 2026 | IBM AML HI-Small | Edge-GraphSAGE AUPRC 0.166 | MLP 0.210, XGBoost 0.172 | Non-graph wins |

The studies that report a large GNN advantage use weak or unequal baselines: Nguyen & Le (LightGBM AUC 0.85 on IEEE-CIS, while the GNN receives label-derived entity features), Chen & Yang (only an MLP on handcrafted features), Gardezi (no baseline details). Expressive GNNs (GINE, PNA) are worse than GraphSAGE in Silva (AP 0.08 and 0.06), while they are the best family on IBM AML in MEGA: the benefit of expressivity seems to depend on whether the labels were *generated* by graph patterns (IBM AML) or not.

This contradicts Cheng et al.'s claim that GNNs "significantly outperform" traditional methods and supports M&R's more cautious view. It is consistent with the original Elliptic study (Weber et al. 2019), where random forests beat GCN. Suggested message for the paper: *in 2024–2026, the evidence for GNN superiority comes from settings where the fraud is defined by graph patterns; on general transaction fraud with strong tabular features, gradient boosting remains at least as good, and the most useful role of the GNN is as an additional signal (fusion) or for pattern- and group-level tasks.*

---

## 4. Results table (input for Shehrazade's table)

Numbers are **not comparable** across rows (different datasets, tasks, metrics).

| Paper | Dataset | Task | Metric | Proposed model | Best baseline |
|---|---|---|---|---|---|
| Bellei 2024 | Elliptic2 | Subgraph classification | Test PR-AUC / ROC-AUC | GLASS 0.208 / 0.889 | GNN-Seg 0.026 / 0.537; Sub2Vec 0.022 / 0.496 |
| Verlaan 2025 | WSM synthetic (10K nodes) | Temporal motif key nodes | AP (GS / SG / C3) | LAS-GNN 1.00 / 0.97 / 1.00 | GIN+Ego+HMP 0.86 / 0.86 / 0.86 |
| Verlaan 2025 | WSM synthetic, balanced | Temporal C6 | F1 | 0.87 | GIN+Ego+HMP 0.49 |
| Bilgi 2026 | IBM AML Medium-HI | Edge classification | Minority F1 | MEGA-PNA 78.26 | Multi-PNA 76.12 |
| Bilgi 2026 | IBM AML Large-LI | Edge classification | Minority F1 | MEGA-PNA 38.31 | Multi-FraudGT 35.49 |
| Bilgi 2026 | ETH phishing | Node classification | Minority F1 | MEGA-PNA 64.84 | Multi-PNA 64.61; DIAM 64.43 |
| Silva 2026 | AMLSim 100k | Edge ranking | AP / R@1% | GraphSAGE 0.783 / 0.877 | LightGBM 0.967 / 0.990 |
| Silva 2026 | AMLSim 1M | Edge ranking | R@1% | GraphSAGE 0.870 | — |
| Nguyen & Le 2025 | IEEE-CIS | Transaction classification | AP / ROC-AUC | HTGNN 2-hop 0.641 / 0.940 | BRIGHT 0.371 / 0.863; LightGBM 0.361 / 0.851 |
| Nguyen & Le 2025 | PaySim | Transaction classification | AP / ROC-AUC | 0.682 / 0.956 | BRIGHT 0.475 / 0.891 |
| Yue 2026 | Elliptic, steps 35–49 | Node classification | AP / F1 | ATGNN-SP 0.262 / 0.265 | Graph HGB 0.262 / 0.271 |
| Chen & Yang 2026 | Taobao (industrial) | Transaction classification | P@20%R | 86.2 | MLP + real-time features 68.5 |
| Chen & Yang 2026 | Offline merchants (industrial) | Transaction classification | P@20%R | 56.5 | 47.2 |
| Dintakurthy 2025 | BankSim 10k | Transaction classification | PR-AUC | Fusion 0.450 (GraphSAGE alone 0.075) | XGBoost 0.425 |
| Sumsuzzaman 2026 | IBM AML HI-Small | Edge classification | AUPRC | 0.166 | MLP 0.210 |
| Gardezi 2025 | Unnamed, pooled | Transaction classification | F1 / AUC | 93.8 / 96.1 | Not reproducible |

---

## 5. Evidence-quality grades

Criteria: peer review and venue; public data; public code; strong baselines; chronological split; size of the test set; consistency of reported numbers.

- **A — MEGA, LAS-GNN, Bellei.** Peer-reviewed in recognized ML/finance venues (TMLR, ICAIF, KDD workshop), public data or code, careful ablations. Limits: LAS-GNN is synthetic only; Bellei is a workshop dataset paper with a random split.
- **B — Silva, Yue, Nguyen & Le.** Sound protocols but narrower evidence: synthetic data (Silva), structure-only features and low-tier venue (Yue), leakage risk and weak baseline (Nguyen & Le).
- **C — Chen & Yang, Dintakurthy, Sumsuzzaman, Gardezi.** Use for trends only, never as evidence of a performance claim. Reasons in §6.

Meta-finding worth stating in the paper: the quality of the recent literature is very uneven. Several 2025–2026 papers in low-selectivity venues propose "frameworks" without reproducible evaluation, which makes surveys that copy their numbers unreliable.

---

## 6. Errors and suspicions

### 6.1 In the papers

**Bellei 2024 (Elliptic2)**
- The headline F1 (0.933 test) is micro-F1, which equals accuracy in single-label classification. From the confusion matrix (TP 245, FN 46, FP 765, TN 11,125): precision 24.3%, recall 84.2%. The PR-AUC (0.208) is the honest number.
- Random 80/10/10 split, no temporal split, although the data span one year.
- Node and edge features were **not used** (memory limits), so the result is structure-only.
- Labels come from a heuristic (illicit → licit path within 6 hops, without change of ownership). A structural model may partly learn the labeling rule. "Suspicious" is not "confirmed laundering".
- Sub2Vec overfits strongly (train PR-AUC 0.145 → test 0.022), not discussed.
- External validation: of the 14 accounts "potentially involved", only 8 are definitively linked to laundering or fraud.

**Verlaan 2025 (LAS-GNN)**
- Ground truth is obtained by exhaustive motif search. An exact algorithm therefore solves the task; the paper does not show what the GNN adds (speed, robustness to noise, generalization to unknown motifs).
- Synthetic graphs only; timestamps are a random permutation; Watts–Strogatz has no hubs.
- Long-cycle results (C4–C6) are on a balanced sample of 2,000 + 2,000 nodes.
- With hubs (LFR, max degree 100) the LSTM aggregator does not learn; memory runs out for lower power-law exponents. Real transaction graphs have hubs.

**Bilgi 2026 (MEGA)**
- Solid. Note: output remains edge-level; ego IDs do not consistently help; the bib entry cites the arXiv version, while the paper is published in TMLR (08/2026).

**Silva 2026**
- Gradient boosting dominates (AP 0.96–0.97 vs 0.78); the GNN is kept "as representative graph-native scorer", which is a design choice, not a result.
- GINE and PNA get AP 0.078 and 0.063, far below the MLP (0.785): possible tuning or implementation problem.
- When a temporal segment has no positives, the split is replaced by a stratified split (contingency mechanism): unclear whether this happened.
- Node features are only 6 aggregates; transaction type is constant.

**Nguyen & Le 2025**
- The abstract says "we expect to demonstrate", wording from a proposal, inconsistent with the results section.
- Entity features include "the total number of recorded fraudulent transactions" of a user, i.e. features derived from labels. Even if limited to past labels, LightGBM does not receive them: unfair comparison and leakage risk.
- LightGBM AUC 0.851 on IEEE-CIS is low compared with well-known public results on this dataset (≈0.92–0.96): weak baseline.
- Only two baselines; 10,000 epochs; the AP row is unlabeled in Table 2.

**Yue 2026**
- The title says "temporal GNN", but the model is a fixed (parameter-free) directed SGC feature expansion followed by an MLP. There is no learned message passing and no temporal graph layer; time enters only through the drift gate.
- The 166 original Elliptic features are discarded on purpose, so absolute scores (F1 0.26) are not comparable with the literature.
- Three seeds only; 15 test snapshots.
- Unrelated works are cited with strong praise in the method section (Audit-HCL "a particularly strong design reference", GT-CSAT "an elegant game-theoretic treatment", TriAudit "demonstrates convincingly"). This looks like citation padding. Audit-HCL is `jiao_dynamic_2026` in our bib: check whether the two papers share authors before relying on either.
- Venue: *Highlights in Business, Economics and Management*, low selectivity. The bib entry has no journal name.
- Positive points: strict prequential protocol, negative result reported explicitly.

**Chen & Yang 2026**
- Both authors work for China Southern Power Grid subsidiaries, yet the paper reports production data and deployment at Ant Financial (Ant Credit Pay Later, Taobao). There is no statement on data access or collaboration with Ant.
- Data availability says "contributions are included in the article": false for the industrial data.
- Table 5: in the offline merchant dataset, the number of buyers and the number of test samples are both 55.85 million: likely a copy error.
- Validation set of 0.2M samples versus 26–42M training/test samples: unusual, not explained.
- C2GAT is described as "originally devised for point-of-interest recommendation": the main module is adapted from earlier work; check originality.
- The t-SNE on Wikipedia is "colored by a proxy for fraudulent vs legitimate behavior": Wikipedia (JODIE) labels are user bans, not fraud.
- Public-data experiments are link prediction and generic node classification, not fraud (the authors admit it).
- Only one baseline on fraud data (MLP + engineered features); no code.

**Dintakurthy 2025**
- 10,000-row extract of BankSim; the test set has 16 frauds, so one fraud changes recall by 6.25 points. The fusion gain (PR-AUC 0.425 → 0.450) is not statistically meaningful.
- Gender appears among the top XGBoost features: fairness problem, not discussed as such.
- Publication date: received Sept. 2025, published 20 Nov. 2025 (vol. 1, special issue 1-2025). The reference given ("17 septembre 2026") and the bib year (2026) are wrong.
- Positive point: standalone GNN kept as an honest negative control.

**Sumsuzzaman 2026**
- The "money-laundering graph" (515,088 nodes, 5,078,345 edges, 0.10% positive) is IBM AML HI-Small, not named as such.
- No comparison with Multi-GNN or MEGA on the same data, which reach minority F1 of about 0.67–0.76 there; AUPRC 0.166 indicates a far weaker model.
- F1, precision and recall are 0 at the default threshold with focal loss.
- The cyber-intrusion half is out of our scope. Writing artifacts in the abstract ("separately78") and text ("lazzy").

**Gardezi 2025**
- The five datasets are not named (e.g. "Credit Card Transactions (Real)", source "Bank Transaction Logs") and cannot be verified.
- Baselines (FinGraphFL, LineMVGNN, FEDGAT-DCNN) are not described; one aggregated result table for all datasets.
- Two paragraphs of the literature review are repeated verbatim (Alenova et al.; Wojcik) in §2.4 and §2.5.
- Federated learning, XAI and cloud–edge deployment are claimed but not evaluated separately. Recommend citing only as an example of low-evidence "framework" papers.

**Li et al. 2025 (review)**
- Ref. [36], *Scalable graph learning for anti-money laundering: A first look* (NeurIPS 2018 workshop), is attributed to Karim et al.; to our knowledge this paper is by Weber et al. (2018). *To verify.*
- AMLSim is called the authors' "proprietary data simulator"; AMLSim is IBM's open-source simulator. *To verify.*

### 6.2 In the report and notes

*Status: all fixes below were applied on 2026-11-15 (main.typ, refs.bib, Notes_Aymen.md).*

- `main.typ:113–116`: "graph pooling operators can compress an entire group of connected nodes into a single representation to flag collective fraud rings" has no citation, and Elliptic2 shows that plain pooling fails (GNN-Seg PR-AUC 0.026) while the labeling trick works (GLASS 0.208). Suggest: cite `bellei_shape_2024` and nuance ("pooling alone is not sufficient; marking subgraph membership and using the surrounding graph is needed").
- `main.typ:189`: "Audit-HCL" and "Subgraph mining" appear in the table without citation. Audit-HCL = `@jiao_dynamic_2026`.
- `main.typ:193`: heading "Existing review papers and  and limits" (repeated "and").
- `main.typ:196`: "ressearch", "developpements".
- `main.typ:218`: "GNN-based financial is overwhelmingly framed…" (missing "fraud detection"); "All three reviews agree;" → colon.
- `main.typ:59` and `refs.bib`: `dintakurthy_multisource_2026` is a 2025 paper (key can stay, year should be 2025).
- `refs.bib`: `bilgi_mega_2026` → cite TMLR 2026; `bellei_shape_2024` → KDD Workshop on Machine Learning in Finance (KDD MLF '24); `yue_adaptive_2026` → add journal *Highlights in Business, Economics and Management*, vol. 69.
- `Notes_Aymen.md`: Elliptic has 203,769 **transaction** nodes, not address nodes (address-level graphs come with Elliptic++). Elliptic2 nodes are **clusters** of addresses. Keep this in mind for the Datasets section (commented table at `main.typ:364–381`).

---

## 7. What remains open (input for Chapter 5)

1. **No real-bank benchmark with group labels.** Elliptic2 is crypto, with heuristic labels; IBM AML and AMLSim are synthetic.
2. **Structuring has no real labels.** Evidence is synthetic (scatter-gather) or qualitative (peeling chains).
3. **Group output is mostly post-hoc.** Only GLASS on Elliptic2 learns at subgraph level; case construction (Silva) loses purity quickly.
4. **Expressive pattern-aware GNNs are tested where patterns define the labels.** Their value on real fraud, where labels come from investigations, is unknown.
5. **GNN vs gradient boosting is unresolved** on strong baselines; fusion seems the most defensible design.
6. **Unsupervised / label-free collective detection** is almost absent from these 10 papers (Audit-HCL is the candidate to read).
7. **Reproducibility** is still low outside the A-grade papers.

---

## 8. Content for the reviews section: Li et al. (2025)

Suggested extra column for `tab:convergences`:

| Theme | Li et al. (2025) |
|---|---|
| Node-level dominance | Taxonomy by learning paradigm; 18 methods in Table 1, mostly node-level; two AML methods detect anomalous subgraphs without GNNs (FlowScope, AntiBenford) |
| Temporal evolution | Dynamic graphs named as a challenge; temporal models covered (GTAN, Tem-GNN, DGANN) |
| Heterogeneous graphs | Mentioned (xFraud, HACUD, MAHINDER) |
| Label scarcity and imbalance | Unsupervised / semi-supervised / supervised taxonomy; PC-GNN for imbalance |
| Dominant backbones | GCN, GAT, GraphSAGE |
| Reference dataset | No benchmark comparison |

Suggested text: *Li et al. (2025) organize graph-based fraud detection by learning paradigm (unsupervised, semi-supervised, supervised) and by application (credit card, insurance, loans, AML). Their Table 1 covers 18 methods, most published before 2023, and the only subgraph-level AML detectors they describe (FlowScope, AntiBenford) are dense-subgraph mining algorithms, not GNNs. They name organized and group-based fraud as a priority but report no benchmark evidence, which confirms the gap identified by the other reviews.*

Note: our table `tab:learning_paradigms` (main.typ:179–191) follows the same paradigm taxonomy as Li et al.; cite it there.

---

## 9. Draft text for Chapter 4 (to adapt)

> **From node scores to fraud structures.** The main advance since the existing reviews is the appearance of models and data that treat laundering as a structure rather than as a set of independent transactions. Elliptic2 provides 122K labeled Bitcoin subgraphs within a 49M-node background graph, and shows that a subgraph method that marks subgraph membership and uses the surrounding graph (GLASS) clearly outperforms pooling node embeddings (PR-AUC 0.208 vs 0.026) @bellei_shape_2024. At the motif level, LAS-GNN extends ego-ID message passing to temporal motifs and detects the key accounts of cycles and scatter-gather schemes in synthetic networks @verlaan_-gnn_2025, while MEGA preserves repeated transfers between the same pair of accounts and improves laundering detection on all IBM AML variants @bilgi_mega_2026. However, these models still output a score per node or per transaction in all banking settings; groups are rebuilt afterwards, at the cost of rapidly decreasing purity @silva_efficient_2026.
>
> **No consistent advantage over tabular models.** When recent studies compare GNNs with gradient boosting under a chronological split, the GNN does not win: tree models dominate on AMLSim @silva_efficient_2026, match the best temporal model on Elliptic @yue_adaptive_2026, and outperform a standalone GraphSAGE on BankSim, where only a fusion of both is best @dintakurthy_multisource_2026. Large GNN gains are reported mainly against weak or unequal baselines @nguyen_real-time_2025 @chen_real-time_2026.

---

## 10. Follow-ups

Papers already in `refs.bib`, not covered here, to read next:
1. `jiao_dynamic_2026` (Audit-HCL, collusive fraud, IBM AML patterns detected weeks in advance): **priority** for the "rings" question. Also check the possible link with Yue et al. (§6.1).
2. `kulkarni_dynberg_2025` (DynBERG), `chen_finfraudbench_2026` (FinFraudBench), `sha_detecting_2025`, `wen_comparative_2026`, `sultana_detectgnn_2025`, `khanvilkar_regulatory_2025` (abstract reports 98.2% F1 on a simulated stream: check), `endah_eka_setiawati_graph_2026`.
