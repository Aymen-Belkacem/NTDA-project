# Synthesis: Cheng et al. (2024) vs. Motie & Raahemi (2024)

Reviews compared:

- **Cheng et al.**: D. Cheng, Y. Zou, S. Xiang, C. Jiang, "Graph neural networks for financial fraud detection: a review", *Frontiers of Computer Science*, 2025 (online 2024). DOI: 10.1007/s11704-024-40474-y. arXiv: 2411.05815. Narrative review, 100+ references.
- **M&R**: S. Motie, B. Raahemi, "Financial fraud detection using graph neural networks: A systematic review", *Expert Systems with Applications*, 240, 122156, 2024. DOI: 10.1016/j.eswa.2023.122156. Systematic review (PRISMA), 33 included papers, literature up to ~mid-2023.

## 1. Convergences (recurring themes in both reviews)

| Theme | M&R (2023, 33 papers) | Cheng et al. (2024, 100+ refs) |
|---|---|---|
| **Node-level dominance** | Almost all studies are node-level; edge-level is rare (< 10%); no paper addresses graph-level anomalies or abnormal communities | 13 methods in Table 3, almost all node-level classification |
| **Dynamic graphs under-explored** | Explicit gap, especially online / streaming settings (50% static, 30% temporal, 20% dynamic) | Dedicated section (§4.3) and temporal architectures (STAGN, STGN), presented as an asset rather than a gap |
| **Heterogeneous graphs** | Needed but under-studied (~30% of papers); mostly node-type heterogeneity only; no cross-domain heterogeneous graphs | HGNN as a full architecture family, with meta-paths and attention |
| **Label scarcity** | Calls for unsupervised / semi-supervised approaches (only one unsupervised paper found) | Addressed through class imbalance (augmentation, cost-sensitive, resampling) and pre-training / self-supervision |
| **Dominant models** | GCN, GraphSAGE, GAT, often in their basic form | GCN and GAT as the foundation |

## 2. Synthesis message

In 2023 (M&R) and in 2024 (Cheng et al.), two independent reviews observe that **collective fraud detection (groups, communities, subgraphs) remains a gap**: GNN-based financial fraud detection is overwhelmingly framed as node-level classification. M&R state explicitly that no reviewed paper covered graph-level anomaly detection or the detection of abnormal communities, although such anomalies ("money laundering or fraudulent groups", §4.2.6) are typical in finance.

> **Positioning of our review:** assess whether the 2024–2026 literature has closed this gap, focusing on GNN architectures for illegal financial rings, structuring, and complex multi-node fraud in transaction networks.

Note on terminology: "graph-level" (classifying a whole graph among a collection of graphs) and "subgraph / community-level" (detecting an anomalous group inside one large graph) are distinct levels. Rings and structuring are primarily subgraph / community-level; they can be cast as graph-level classification by extracting candidate subgraphs first.

## 3. Critical points to keep (for our own methodology)

1. **Search string misses AML (anti-money laundering) work.** M&R's fraud concept block is `fraud OR outlier OR anomaly`, with no *laundering*, *illicit* or *AML*. AML papers that do not use these three words were likely missed (e.g., LaundroGraph, Cardoso et al. 2022, cited by Cheng et al., is absent). Weber et al. 2019 (Elliptic) is also absent, possibly excluded as a workshop paper. Our query should add at least: *laundering, illicit, AML, smurfing, structuring, collusion, ring, group, community*.
2. **Uneven queries across databases.** On ScienceDirect, M&R's finance block was reduced to `credit OR tax OR insurance` (no crypto, no bank).
3. **Preprints excluded.** M&R acknowledge it themselves: a strong limitation in a field where much work first appears on arXiv.
4. **Laundering motifs mentioned, not demonstrated.** M&R §4.1 describes cycles, fan-in / fan-out flows and account clustering as things GNNs *can* detect, but as an illustrative example not supported by any of the 33 reviewed papers. This reinforces the gap.
5. **Artificially rebalanced evaluations.** Several papers rebalance data to 50% fraud, which inflates results. Argues for requiring imbalance-aware metrics (e.g., AUC-PR) in our analysis grid.
6. **No quantitative comparison in either review.** Cheng et al. claim GNNs "significantly outperform" traditional methods; M&R are more nuanced (traditional ML can remain competitive on small or weakly structured graphs). Neither provides a benchmark table.
7. **Low reproducibility.** Only ~15% of the papers reviewed by M&R release their code.
