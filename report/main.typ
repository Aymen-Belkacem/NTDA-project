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


To uncover coordinated fraud, Graph Neural Networks (GNNs) first organize raw
transaction records into a directed network, formally written as a graph
$cal(G) = (cal(V), cal(E))$. In this
network, the nodes $cal(V)$ represent the actors in the financial system—such
as bank accounts, credit cards, merchants, or devices—while the directed edges
$cal(E)$ represent the interactions between them, such as money transfers or
shared devices. Because different
actors and transactions carry different meanings, this graph is heterogeneous
and attributed: each node $v$ has its own profile vector $bold(x)_v$ (e.g.,
account age or balance), and each edge $(u, v)$ carries transaction details
$bold(x)_(u v)$ (e.g., transfer amount and timestamp).

Once the graph is built, a GNN evaluates whether an account or transaction is
suspicious through an iterative process called message passing.
The intuition is simple: an
account's risk depends not only on its own activity, but also on who it
interacts with. At each step (or layer) $k$, every node $v$ gathers information
from its immediate neighbors $cal(N)(v)$, aggregates those signals using a
simple operation like a sum or average, and updates its own internal risk
representation $bold(h)_v^((k))$:
$ bold(h)_v^((k)) = "UPDATE"^((k)) ( bold(h)_v^((k-1)), plus.o_(u in cal(N)(v)) "MSG"^((k))(bold(h)_u^((k-1)), bold(h)_v^((k-1)), bold(x)_(u v)) ) $
Here, $"MSG"$ extracts the relevant signal from each neighbor $u$ and edge
$bold(x)_(u v)$, $plus.o$ combines all incoming messages regardless of their
order, and $"UPDATE"$ merges this neighborhood context with the node's previous
state $bold(h)_v^((k-1))$. Repeating this process across $K$ layers allows
information to travel $K$ steps across the network, revealing indirect money
laundering chains, while graph pooling operators can compress an entire group
of connected nodes into a single representation to flag collective fraud rings.

#figure(
  block(
    fill: luma(248),
    stroke: 0.6pt + luma(180),
    radius: 4pt,
    inset: 10pt,
    width: 100%,
  )[
    #set text(size: 8.5pt)
    *Raw Transactions*
    $arrow.r$ *Graph* $cal(G)$ (Entities $cal(V)$ & Transfers $cal(E)$)
    $arrow.r$ *K-Hop Message Passing* (GCN / GraphSAGE / GAT)
    $arrow.r$ *Risk Score* $hat(y)_v in [0, 1]$
  ],
  caption: [From raw financial transactions to graph-based fraud risk scoring.],
) <fig:mpnn_pipeline>

Building on this message passing principle, three foundational architectures
have shaped financial fraud detection, each addressing a limitation of its
predecessor. First, Graph Convolutional
Networks (GCNs) compute a normalized average of all neighbor features using
node degrees, providing a clean baseline on static graphs like Bitcoin;
however, GCN requires the entire graph during training
and cannot easily handle newly created accounts. To overcome this, GraphSAGE
introduces inductive learning by sampling a fixed-size neighborhood and
combining a node's own features with its sampled neighbors, allowing banks to
score unseen accounts in real time. Yet, both GCN and GraphSAGE treat every
neighbor with equal structural weight, which becomes a weakness when fraudsters
hide among hundreds of normal clients. Graph Attention Networks (GATs) solve
this issue by learning dynamic attention weights $alpha_(u v)$ that focus on
suspicious links while ignoring benign noise.

#figure(
  caption: [Progression of foundational GNN architectures for financial fraud detection.],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Model][Core Idea][Key Advantage][Main Limitation in Fraud],
    [*GCN*], [Degree-normalized average of all neighbors], [Simple and effective on static networks], [Transductive (cannot generalize to unseen accounts)],
    [*GraphSAGE*], [Fixed-size neighbor sampling + self-concatenation], [Inductive scaling to new accounts and merchants], [Treats benign and suspicious neighbors equally],
    [*GAT*], [Learnable attention weights $alpha_(u v)$ on edges], [Filters out noisy benign neighbors], [Higher memory and latency cost on dense graphs],
  )
) <tab:gnn_backbones>

Finally, how these architectures are trained depends heavily on label
availability in real banking systems, where confirmed fraud cases are rare and
often take months to investigate. When historical audit
labels are available, supervised learning trains GNNs directly with weighted or
focal classification losses to handle extreme class imbalance.
Because annotating millions of
transactions is too costly, semi-supervised learning exploits the graph
structure itself to propagate risk signals from a small labeled subset to the
surrounding unlabeled network. Conversely, when no
confirmed fraud labels exist at all—such as in emerging laundering schemes—
unsupervised and self-supervised methods detect structural outliers by
reconstructing graph attributes, mining dense subgraphs,
or aligning structural and temporal views via contrastive
learning.

#figure(
  caption: [Comparison of GNN training paradigms according to label availability.],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Paradigm][When to Use It][How It Learns][Key Examples],
    [*Supervised*], [Verified historical fraud labels are available], [Minimizes imbalance-aware classification loss], [GCN, PC-GNN],
    [*Semi-Supervised*], [Only a small fraction of transactions is labeled], [Propagates risk from labeled nodes to unlabeled neighbors], [GTAN],
    [*Unsupervised*], [Zero fraud labels or unknown new attack patterns], [Detects abnormal subgraphs or contrastive mismatches], [Subgraph mining, Audit-HCL],
  )
) <tab:learning_paradigms>

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

== Technical foundations <sec:foundations>

Standard message passing has theoretical limits: some graph problems cannot
be solved by it on the given input graph @velickovic_message_2022. This
matters in fraud detection, because laundering patterns are often small
directed subgraphs. Simple adaptations such as port numbering, ego IDs, and
reverse message passing allow a standard GNN to detect any directed subgraph
pattern and improve laundering detection @egressy_provably_2024. This link
between fraud and subgraph shapes also explains why some work classifies whole
subgraphs instead of single nodes @bellei_shape_2024 @liu_graph_2023.

Financial graphs are rarely simple. They contain several node and edge types,
so message passing must be extended to heterogeneous graphs, with a dedicated
way to aggregate messages across edge types @johannessen_finding_2023. Two
accounts may also be linked by many transfers, and neighbor-aware aggregation
keeps the information of repeated interactions @bilgi_mega_2026. Moreover,
graphs evolve: models therefore work on sequences of snapshots or use temporal
attention and recurrent units to follow the changes
@cheng_graph_2022 @kulkarni_dynberg_2025 @yue_adaptive_2026. A related
precaution is to avoid future information during message passing and to split
data chronologically @lu_bright_2022 @yue_adaptive_2026.

Three further issues are specific to fraud. First, deep aggregation can
over-smooth representations and mix suspicious nodes with their benign
neighbors, which high-order representation learning tries to avoid
@zou_effective_2024. Second, camouflage hides fraudsters, both in their
features and in their relations, and camouflage-resistant selection of
neighbors was designed against it @dou_enhancing_2020. From a spectral point
of view, anomalies shift the energy toward high frequencies, which motivates
band-pass filters @tang_rethinking_nodate. Third, fraud labels are extremely
imbalanced, so models rely on dedicated sampling and loss functions
@liu_pick_2021 @sha_detecting_2025. For the same reason, evaluation uses
imbalance-sensitive measures such as F1-score, average precision, or PR-AUC
rather than accuracy @yue_adaptive_2026 @endah_eka_setiawati_graph_2026
@chen_finfraudbench_2026. @tab:tech_challenges summarizes these points.

#figure(
  caption: [Technical challenges of GNNs for fraud detection and representative remedies.],
  placement: top,
  scope: "parent",
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Challenge][Why It Matters in Fraud][Representative Remedy][Examples],
    [*Limited expressivity*], [Laundering patterns are directed subgraphs], [Port numbers, ego IDs, reverse messages], [@egressy_provably_2024 @velickovic_message_2022],
    [*Rich graph types*], [Many entity types and repeated transfers], [Heterogeneous and multigraph aggregation], [@johannessen_finding_2023 @bilgi_mega_2026],
    [*Time dynamics*], [Behavior and tactics change], [Snapshots, temporal attention, recurrent units], [@cheng_graph_2022 @kulkarni_dynberg_2025],
    [*Over-smoothing*], [Fraudsters blend with normal neighbors], [High-order pure representations], [@zou_effective_2024],
    [*Camouflage*], [Fraudsters imitate normal users], [Careful neighbor selection, spectral filters], [@dou_enhancing_2020 @tang_rethinking_nodate],
    [*Class imbalance*], [Fraud is extremely rare], [Imbalance-aware sampling and losses], [@liu_pick_2021 @sha_detecting_2025],
  ),
) <tab:tech_challenges>



== Financial Fraud

Financial fraud covers a wide family of deliberate schemes designed to obtain
money illegally, and classic surveys organize it by application domain and by
detection technique @abdallah_fraud_2016 @ahmed_survey_2016
@irofti_quick_2019. Graph-based surveys focus on three main scenarios: credit
card fraud, insurance fraud, and money laundering @li_graph_2025. We add two
further settings that appear often in the GNN literature, namely e-commerce
transaction fraud and phishing on blockchains, and summarize all of them in
@tab:fraud_scenarios.

In credit card fraud, criminals disguise illegal payments by routing them
through several honest users, which makes the fraud hard to see in a single
transaction @zou_effective_2024. In anti-money laundering (AML), banks must
monitor huge volumes of transfers, while rule-based systems remain imprecise
and criminals often work together
@johannessen_finding_2023 @deprez_advances_2025. The cost of this regulation
and the openness of cryptocurrencies also made Bitcoin a popular testbed for
forensic analysis @weber_anti-money_2019. Phishing on Ethereum can likewise be
treated as a graph problem, where addresses and their transfers form the
network @wang_phishing_2022. Finally, e-commerce platforms must score incoming
transactions among many different entity types, at very large scale
@rao_xfraud_2021 @lu_bright_2022.

#figure(
  caption: [Main financial fraud scenarios studied with graph learning.],
  placement: top,
  scope: "parent",
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Scenario][Graph Structure][Main Difficulty][Examples],
    [*Credit card fraud*], [Cardholders, merchants and transactions], [Fraud hidden behind legitimate transactions, scarce labels], [@zou_effective_2024 @xiang_semi-supervised_2023],
    [*Money laundering (AML)*], [Accounts and transfers, in banks or on blockchains], [Cooperating criminals, late labels, weak rule-based systems], [@johannessen_finding_2023 @weber_anti-money_2019],
    [*Phishing on blockchain*], [Addresses and transactions on Ethereum], [Imitation of normal addresses], [@wang_phishing_2022 @egressy_provably_2024],
    [*E-commerce fraud*], [Many typed entities in transaction logs], [Scale, low latency, need for explanations], [@rao_xfraud_2021 @lu_bright_2022],
  ),
) <tab:fraud_scenarios>

Across these scenarios, four obstacles keep coming back. Fraudsters imitate
normal behavior to escape detection @dou_enhancing_2020, fraud examples are
extremely rare compared with normal ones @liu_pick_2021, criminals constantly
adapt their tactics, which forces models to keep learning over time
@deprez_advances_2025, and banks need fast decisions together with
explanations that regulators can audit @rao_xfraud_2021
@khanvilkar_regulatory_2025. In collusion rings, confirmed labels may even
arrive months after the facts @jiao_dynamic_2026. These obstacles motivate the
technical choices described in the next section.

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
