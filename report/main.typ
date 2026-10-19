#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Graph Neural Networks for Financial Fraud Detection: A Review],
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
Graph neural networks (GNNs) have become a prominent approach for financial fraud
detection. A recent systematic review @motie2024financial analysed 33 articles and
called for further work on dynamic and heterogeneous graphs, particularly in
cross-domain applications.




= Review Methodology <sec:method>
// TODO: research questions, databases searched, search query, inclusion/exclusion
// criteria, PRISMA-style selection flow (records found -> screened -> included).
#lorem(80)

= Existing review papers and  and limits 
// TODO
= Background <sec:background>

== Technical foundations of Graph Neural Networks
// TODO: message passing, GCN, GAT, heterogeneous GNNs, graph transformers.
#lorem(60)

== Financial Fraud
// TODO: fraud types (money laundering, credit card fraud, insurance, tax evasion...).
#lorem(60)

= Taxonomy of GNN-Based Approaches <sec:taxonomy>
Following @motie2024financial, existing models can be grouped into four families
(@fig:taxonomy): convolutional-based, attention-based, metapath-based and
transformer-based models.

#figure(
  image("figures/motie2024-taxonomy.jpg", width: 100%),
  caption: [Taxonomy of GNN models for financial fraud detection, from @motie2024financial.],
) <fig:taxonomy>

== Convolutional-Based Models
#lorem(50)

== Attention-Based Models
#lorem(50)

== Metapath-Based Models
#lorem(50)

== Transformer-Based Models
#lorem(50)

= Graph Types <sec:graphs>
// TODO: homogeneous vs. heterogeneous; static vs. temporal vs. dynamic graphs.
#lorem(80)

= Datasets <sec:datasets>
@tab:datasets summarises the main public graph datasets for financial fraud detection.

#figure(
  caption: [Public graph datasets for financial fraud detection],
  placement: top,
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[Dataset][Nodes][Edges],
    [Elliptic], [203,769], [234,355],
    [Elliptic2 @bellei2024shape], [49M], [196M],
    [T-Finance], [39,357], [21,222,543],
    [DGraph], [TODO], [TODO],
    [FinFraudBench], [TODO], [TODO],
  )
) <tab:datasets>

// TODO: describe each dataset (domain, labels, anomaly ratio, graph type).
#lorem(80)

= Recent developpements not covered  by existing review papers

= Discussions 


= Open Challenges and Future Directions <sec:challenges>
// TODO: edge- and graph-level anomaly detection; unsupervised and semi-supervised
// learning; dataset scarcity (taxation, real estate, mortgage); dynamic graphs.
#lorem(80)

= Conclusion
#lorem(60)
