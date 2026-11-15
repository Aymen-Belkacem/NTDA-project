# Synthesis of existing reviews: Cheng et al. (2024), Motie & Raahemi (2024), Deprez et al. (2025)

Reviews compared:

- **Cheng et al.**: D. Cheng, Y. Zou, S. Xiang, C. Jiang, "Graph neural networks for financial fraud detection: a review", *Frontiers of Computer Science*, 2025 (online 2024). DOI: 10.1007/s11704-024-40474-y. arXiv: 2411.05815. Narrative review, 100+ references.
- **M&R**: S. Motie, B. Raahemi, "Financial fraud detection using graph neural networks: A systematic review", *Expert Systems with Applications*, 240, 122156, 2024. DOI: 10.1016/j.eswa.2023.122156. Systematic review (PRISMA), 33 included papers, literature up to ~mid-2023.
- **Deprez et al.**: B. Deprez, W. Wei, W. Verbeke, B. Baesens, K. Mets, T. Verdonck, "Advances in Continual Graph Learning for Anti-Money Laundering Systems: A Comprehensive Review", *WIREs Computational Statistics*, 17(3), e70040, September 2025. DOI: 10.1002/wics.70040. arXiv: 2503.24259 (preprint v1, March 2025, was the version read). Peer-reviewed. Short literature review (~6 GNN-for-AML papers, ~5 continual-learning-for-fraud papers) plus an experimental benchmark (code public). AML = anti-money laundering.

## 1. Convergences (recurring themes across reviews)

| Theme | M&R (2023, 33 papers) | Cheng et al. (2024, 100+ refs) | Deprez et al. (2025, WIREs) |
|---|---|---|---|
| **Node-level dominance** | Almost all studies are node-level; edge-level is rare (< 10%); no paper addresses graph-level anomalies or abnormal communities | 13 methods in Table 3, almost all node-level classification | Node classification (Elliptic) **and** edge classification (IBM AML); still no subgraph-level detection |
| **Fraud evolves over time** | Explicit gap: dynamic graphs, online / streaming settings (50% static, 30% temporal, 20% dynamic) | Dedicated section (§4.3) and temporal architectures (STAGN, STGN); RL for adaptivity as future direction | **Core topic**: catastrophic forgetting when fine-tuning; addressed with continual learning |
| **Heterogeneous graphs** | Needed but under-studied (~30% of papers); mostly node-type heterogeneity only; no cross-domain heterogeneous graphs | HGNN as a full architecture family, with meta-paths and attention | Homogeneous graphs only (one reviewed work, HTG-CFD, is heterogeneous) |
| **Label scarcity / class imbalance** | Calls for unsupervised / semi-supervised approaches (only one unsupervised paper found); 45% of datasets have < 2% anomalies | Addressed through class imbalance (augmentation, cost-sensitive, resampling) and pre-training / self-supervision | 0.1% laundering in IBM AML; the jointly trained model collapses to the majority class |
| **Dominant models** | GCN, GraphSAGE, GAT, often in their basic form | GCN and GAT as the foundation | GCN and GAT are the usual backbones; only GCN is tested |
| **Elliptic as reference dataset** | Used by 7 of the 25 papers with public data | Cited (Weber et al.) | Used (node classification, 49 time steps) |
| **Insurance** | 2 papers (medical insurance) | Organized fraud via risk diffusion | POCL (continual learning for medical insurance fraud) |

## 2. Synthesis message

In 2023 (M&R), 2024 (Cheng et al.) and 2025 (Deprez et al.), the reviews converge: **collective fraud detection (groups, communities, subgraphs) remains a gap**. GNN-based financial fraud detection is overwhelmingly framed as node-level (and occasionally edge-level) classification. M&R state explicitly that no reviewed paper covered graph-level anomaly detection or the detection of abnormal communities, although such anomalies ("money laundering or fraudulent groups", §4.2.6) are typical in finance.

Deprez et al. are the first of the three to operationalize **money laundering typologies** as labels (the 8 patterns of the IBM AML dataset: fan-in, fan-out, bipartite, stack, cycle, random, scatter-gather, gather-scatter). However, each transaction is still classified **individually** with its pattern label: the model never outputs a ring or a pattern as an object. In 2025, multi-node pattern detection is therefore still approached through an **edge-level proxy**, not through genuine subgraph detection.

Mapping to our research question:
- **Rings** ↔ *cycle*
- **Structuring / smurfing** (approximately) ↔ *fan-out*, *scatter-gather*
- **Complex multi-node fraud** ↔ *bipartite*, *stack*, *gather-scatter*

> **Positioning of our review:** assess whether the 2024–2026 literature has closed this gap, focusing on GNN architectures for illegal financial rings, structuring, and complex multi-node fraud in transaction networks (banking, crypto, insurance).

Note on terminology: "graph-level" (classifying a whole graph among a collection of graphs) and "subgraph / community-level" (detecting an anomalous group inside one large graph) are distinct levels. Rings and structuring are primarily subgraph / community-level; they can be cast as graph-level classification by extracting candidate subgraphs first.

## 3. Two distinct temporal axes

- **Time in the data** (Cheng et al., M&R): temporal / dynamic graphs, temporal attention, GNN + RNN, time-decay functions.
- **Time in the model lifecycle** (Deprez et al.): updating a deployed model on new data without forgetting older laundering tactics (continual learning: replay, regularization, architecture-based). Also driven by regulation (limits on how long transaction data can be stored) and by compute budgets.

For our review, continual learning is a deployment concern rather than an architecture; treat it as a secondary dimension ("temporal adaptation").

## 4. What Deprez et al. add for our research question

1. **Explicit laundering typologies**: placement / layering / integration, and the 8 IBM AML patterns (Altman et al., 2023). Cheng et al. cite this dataset only as a graph-construction example; M&R predate it.
2. **IBM AML is the reference benchmark for pattern-level work**, since it carries pattern labels. Elliptic has no pattern labels and its labels are "illicit", not specifically laundering.
3. **Depth vs. chain length**: the only review linking an architectural choice (number of layers) to the reach of multi-hop laundering chains, traded off against over-smoothing and forgetting.
4. **Vanilla GCN fails on IBM patterns** (average micro-F1 ~0.12–0.21; final performance as low as 0.0001 in several settings). This is consistent with Egressy et al. (*Provably Powerful GNNs for Directed Multigraphs*), who show that standard message-passing GNNs cannot capture some patterns (e.g., cycles). Deprez et al. cite Egressy et al. only to order patterns by difficulty. **Pattern-aware architectures (reverse message passing, multigraph port numbering, ego IDs) are likely the core of the state of the art we are looking for; to verify in the 2024–2026 literature.**
5. **Results**: replay (GEM) gives the best performance with the least forgetting; more epochs or wider models lead to more (and sudden) forgetting; task order has no significant effect.

## 5. Critical points to keep (for our own methodology)

1. **Search string misses AML work.** M&R's fraud concept block is `fraud OR outlier OR anomaly`, with no *laundering*, *illicit* or *AML*. AML papers that do not use these three words were likely missed (e.g., LaundroGraph, Cardoso et al. 2022, cited by Cheng et al. and Deprez et al., is absent). Weber et al. 2019 (Elliptic) is also absent, possibly excluded as a workshop paper. Our query should add at least: *laundering, illicit, AML, smurfing, structuring, collusion, ring, group, community, pattern, motif, cycle*.
2. **Uneven queries across databases.** On ScienceDirect, M&R's finance block was reduced to `credit OR tax OR insurance` (no crypto, no bank).
3. **Preprints excluded.** M&R acknowledge it themselves: a strong limitation in a field where much work first appears on arXiv (Deprez et al. appeared on arXiv six months before its journal publication).
4. **Laundering motifs mentioned, not demonstrated.** M&R §4.1 describes cycles, fan-in / fan-out flows and account clustering as things GNNs *can* detect, but as an illustrative example not supported by any of the 33 reviewed papers. This reinforces the gap.
5. **Inappropriate metrics under imbalance.** Several papers rebalance data to 50% fraud, which inflates results (M&R). Deprez et al. choose micro-F1 "because of imbalance", but in single-label classification micro-F1 equals accuracy. Our analysis grid should require imbalance-aware metrics (macro-F1, AUC-PR).
6. **Discarded labels.** Deprez et al. drop the 1,968 laundering transactions without a pattern label (38% of the 5,177 laundering transactions in HI-Small).
7. **No cross-method benchmark in the narrative reviews.** Cheng et al. claim GNNs "significantly outperform" traditional methods; M&R are more nuanced (traditional ML can remain competitive on small or weakly structured graphs). Neither provides a benchmark table. Deprez et al. do run experiments, but with a single backbone (GCN) on two datasets.
8. **Low reproducibility.** Only ~15% of the papers reviewed by M&R release their code (Deprez et al. do release theirs).
9. **Misleading titles.** Deprez et al.'s "Comprehensive Review" is mostly an experimental paper preceded by a review of about ten works.
