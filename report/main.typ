#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A state of the art of GNN architectures applied to transaction networks to identify illegal financial rings, structuring, and complex multi-node fraud],
  abstract: [
    // TODO: write the abstract last (150–250 words): context, scope of the review,
    // method (databases, number of papers), main findings, open challenges.
    #lorem(60)
  ],
  authors: (
    (
      name: "Aymen Belkacem",
      department: [TODO: Department],
      organization: [TODO: University],
      location: [TODO: City, Country],
      email: "aymen.belkacem@gmail.com"
    ),
    // Add co-authors here with the same structure.
  ),
  index-terms: (
    "Graph neural networks",
    "Fraud detection",
    "Graph anomaly detection",
    "Financial transactions",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#set text(lang: "en")

= Introduction


As financial services move online—from instant mobile payments to
cryptocurrencies—financial fraud has evolved from isolated incidents into
highly coordinated criminal networks @abdallah_fraud_2016 @ahmed_survey_2016.
Today, fraudsters split stolen funds across dozens of accounts, share devices,
and route money through complex chains of intermediaries to stay invisible.
Traditional defense systems, such as manual rules and classic machine learning
models (like Random Forests or XGBoost), struggle to catch these schemes
because they inspect each transaction in isolation, like looking at a single
puzzle piece without seeing the whole picture @weber_anti-money_2019
@pei_subgraph_2020. Graph Neural Networks (GNNs) change this perspective by
connecting the dots @scarselli_graph_2009 @wu_graph_2022: by mapping accounts,
merchants, and devices as nodes and transfers as links, GNNs naturally uncover
hidden collusion rings that single-transaction models miss @li_graph_2025.

However, applying GNNs to real-world banking still faces major hurdles.
Fraudsters deliberately camouflage themselves by interacting with normal users
@dou_enhancing_2020, fraudulent transactions remain extremely rare
@liu_pick_2021, criminal tactics evolve rapidly over time @yue_adaptive_2026,
and banks need fast decisions alongside clear explanations for regulators
@lu_bright_2022 @rao_xfraud_2021. While recent surveys catalog existing GNN
fraud detectors @motie_financial_2024 @cheng_graph_2025 @li_graph_2025, they
rarely highlight what GNNs still cannot achieve alone. In this review, we go
one step further by examining these practical limits and showcasing emerging
solutions beyond standard GNNs—such as combining GNNs with tabular models
@dintakurthy_multisource_2026, temporal graph transformers
@kulkarni_dynberg_2025 @chen_real-time_2026, continual learning
@deprez_advances_2025, and Large Language Models (LLMs) to explain alerts to
regulators @khanvilkar_regulatory_2025.

Following our review structure, this paper addresses three core Research
Questions (RQs):
- *RQ1 (Methodology & Taxonomy):* How can existing GNN approaches for
  financial fraud detection be organized into a clear taxonomy, and what
  benchmark datasets are used to evaluate them?
- *RQ2 (Results):* How do different GNN models compare in practice on public
  benchmarks, and what gaps remain in current evaluation protocols?
- *RQ3 (Discussion — Beyond GNNs):* What are the main limits of current GNNs,
  and what next-generation approaches can go beyond them to improve real-world
  fraud detection?

The remainder of this paper is organized as follows. (A REVOIR A LA FIN)





= Methodology <sec:method>
// TODO: research questions, databases searched, search query, inclusion/exclusion
// criteria, PRISMA-style selection flow (records found -> screened -> included).
#lorem(80)

== Background <sec:background>

https://github.com/Aymen-Belkacem/NTDA-project

A faire par Shehrazade

== Existing review papers and  and limits 
// TODO A faire par aymen, bientot fait

Multiple research papers with similar ressearch questions have been published in the last 3 years @motie_financial_2024 @cheng_graph_2025 @deprez_advances_2025. In this section we synthesize their findings to establish a foundation for new developpements in the subject of fraud detection. Recurring themes and convergences across the reviews are summarized in @tab:convergences.

#figure(
  caption: [Recurring themes across existing reviews],
  placement: top,
  scope: "parent",
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },

    table.header[Theme][Motie & Raahemi @motie_financial_2024][Cheng et al. @cheng_graph_2025][Deprez et al. @deprez_advances_2025],
    [Node-level dominance], [Nearly all node-level; edge-level under 10%; no graph- or community-level work], [Nearly all surveyed methods are node-level], [Node (Elliptic) and edge (IBM AML) classification; no subgraph-level detection],
    [Temporal evolution], [Gap: dynamic and online graphs (50% of studies use static graphs)], [Temporal GNNs (e.g., STAGN, STGN)], [Core topic: continual learning against catastrophic forgetting],
    [Heterogeneous graphs], [Under-studied (about 30% of studies)], [Dedicated HGNN family (meta-paths, attention)], [Homogeneous graphs only],
    [Label scarcity and imbalance], [Calls for unsupervised and semi-supervised methods], [Resampling, cost-sensitive learning, augmentation, pre-training], [0.1% laundering (IBM AML); joint training collapses to the majority class],
    [Dominant backbones], [GCN, GraphSAGE, GAT], [GCN, GAT], [GCN (only backbone tested)],
    [Reference dataset], [Elliptic (7 of 25 papers with public data)], [Elliptic], [Elliptic, IBM AML],
  )
) <tab:convergences>

All three reviews agree; collective fraud detection (groups, communities, subgraphs) remains a gap. GNN-based financial is overwhelmingly framed as node-level and, occasionally, edge-level detection. Motie and Raahemi @motie_financial_2024 state explicitly that no reviewed paper covered graph-level anomaly detection or the detection of abnormal communities, although such anomalies are typical in finance.

Deprez et al. @deprez_advances_2025 operationalize money laundering typologies as labels (the 8 patterns of the IBM AML dataset: fan-in, fan-out, bipartite, stack, cycle, random, scatter-gather, gather-scatter). However, each transaction is still classified individually with its pattern label, and the model never outputs a ring or a pattern as an object. In 2025, multi-node pattern detection is therefore still approached through an edge-level proxy, not through genuine subgraph detection.



== Technical foundations of Graph Neural Networks
// TODO: message passing, GCN, GAT, heterogeneous GNNs, graph transformers.
#lorem(60)

A faire par Shehrazade

== Financial Fraud
// TODO: fraud types (money laundering, credit card fraud, insurance, tax evasion...).
#lorem(60)

A faire par Shehrazade

== Taxonomy of GNN-Based Approaches <sec:taxonomy>
// Following @motie2024financial, existing models can be grouped into four families
// (@fig:taxonomy): convolutional-based, attention-based, metapath-based and
// transformer-based models.

// #figure(
//   image("figures/motie2024-taxonomy.jpg", width: 100%),
//   caption: [Taxonomy of GNN models for financial fraud detection, from @motie2024financial.],
// ) <fig:taxonomy>

A faire par Aymen

=== Convolutional-Based Models
#lorem(50)
A faire par Aymen

=== Attention-Based Models
#lorem(50)
A faire par Aymen

=== Metapath-Based Models
#lorem(50)
A faire par Aymen

=== Transformer-Based Models
#lorem(50)
A faire par Aymen


= Datasets <sec:datasets>
// @tab:datasets summarises the main public graph datasets for financial fraud detection.

A faire par Aymen

// #figure(
//   caption: [Public graph datasets for financial fraud detection],
//   placement: top,
//   table(
//     columns: (auto, auto, auto),
//     align: (left, right, right),
//     inset: (x: 6pt, y: 4pt),
//     stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
//     fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

//     table.header[Dataset][Nodes][Edges],
//     [Elliptic], [203,769], [234,355],
//     [Elliptic2 @bellei2024shape], [49M], [196M],
//     [T-Finance], [39,357], [21,222,543],
//     [DGraph], [TODO], [TODO],
//     [FinFraudBench], [TODO], [TODO],
//   )
// ) <tab:datasets>

// TODO: describe each dataset (domain, labels, anomaly ratio, graph type).
#lorem(80)

= Results

A faire par Shehrazade et Aymen

Tableau de resultats: Shehrazade


= Open Challenges and Future Directions <sec:challenges>
// TODO: edge- and graph-level anomaly detection; unsupervised and semi-supervised
// learning; dataset scarcity (taxation, real estate, mortgage); dynamic graphs.
#lorem(80)

A faire par 

= Discussions and conclusion

A faire par Aymen et Shehrazade
