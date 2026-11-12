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
// TODO: motivation (cost of financial fraud), why graphs, contributions of this review.

A faire par Shehrazade

Graph neural networks (GNNs) have become a prominent approach for financial fraud
detection. A recent systematic review analysed 33 articles and
called for further work on dynamic and heterogeneous graphs, particularly in
cross-domain applications.




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