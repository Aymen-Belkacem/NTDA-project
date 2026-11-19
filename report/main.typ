#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1/1")

#show: ieee.with(
  title: [A state of the art of GNN architectures applied to transaction networks to identify illegal financial rings, structuring, and complex multi-node fraud],
  abstract: abstract: [Financial fraud and money laundering have evolved from isolated incidents into coordinated multi-entity operations—such as layering chains, cycles, and collusion rings—that routinely evade traditional rule-based systems and tabular classifiers. While machine learning excels on independent transactions, financial data naturally form complex, non-Euclidean transaction graphs characterized by intricate relational dependencies and rapid temporal evolution. Recently, Graph Neural Networks (GNNs) have emerged as a leading paradigm for learning representations on financial networks. In this survey, we provide a comprehensive state-of-the-art review of GNN architectures applied to transaction networks for financial crime detection. We organize existing models into a unified taxonomy spanning convolutional, attention-based, meta-path, transformer, and subgraph-aware message-passing paradigms. We further examine 11 public benchmark datasets across cryptocurrency, synthetic banking, and tabular domains, and synthesize empirical findings from 9 recent studies (2024–2026). Our analysis highlights critical evaluation gaps: despite advances in temporal and subgraph modeling, multi-node rings are still largely reconstructed post hoc, and standalone GNNs rarely outperform gradient-boosted trees under strict chronological splits. Finally, we outline key open challenges and future research directions, including label-free collective detection, hybrid graph–tabular fusion, and privacy-preserving real-world banking benchmarks.
  ],
  authors: (
    (
      name: "Aymen Belkacem",
      department: [TODO: Department],
      organization: [HES-SO],
      location: [Lausanne, Switzerland],
      email: "aymen.belkacem@hes-so.ch"
    ),
    (
      name: "Sherazade Ligali",
      department: [TODO: Department],
      organization: [HES-SO],
      location: [Lausanne, Switzerland],
      email: "sherazade.ligali@hes-so.ch"
    ),
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
Traditional defense systems rely on manual rules or classic tabular machine
learning models (such as Random Forests or XGBoost) that evaluate individual
transactions or accounts separately @weber_anti-money_2019 @pei_subgraph_2020.
While well-tuned tabular models remain strong baselines when rich engineered
features are available @weber_anti-money_2019, modeling financial activity as a
graph offers a complementary structural perspective @scarselli_graph_2009
@wu_graph_2022: by representing accounts, merchants, and devices as nodes and
transfers as links, Graph Neural Networks (GNNs) aim to capture relational
patterns and multi-hop dependencies that purely tabular representations can
overlook @li_graph_2025.

However, applying GNNs to real-world banking still faces major hurdles.
Fraudsters deliberately camouflage themselves by interacting with normal users
@dou_enhancing_2020, fraudulent transactions remain extremely rare
@liu_pick_2021, criminal tactics evolve rapidly over time @yue_adaptive_2026,
and banks require low-latency decisions alongside clear explanations for
regulators @lu_bright_2022 @rao_xfraud_2021. Moreover, most deployed GNNs still
score individual nodes or edges rather than directly outputting complete fraud
rings, requiring post-hoc group reconstruction. While recent surveys catalog
existing GNN fraud detectors @motie_financial_2024 @cheng_graph_2025
@li_graph_2025, they rarely examine when GNNs actually outperform strong
tabular baselines or what architectural and evaluation gaps remain. In this
review, we go one step further by analyzing recent advances in subgraph and
temporal modeling, evaluating how GNNs compare against gradient-boosted trees
under realistic protocols, and highlighting emerging hybrid directions—such as
fusing GNNs with tabular models @dintakurthy_multisource_2026 and temporal
graph attention @kulkarni_dynberg_2025 @chen_real-time_2026.

Following our review structure, this paper addresses three core Research
Questions (RQs):
- *RQ1 (Methodology & Taxonomy):* How can existing GNN approaches for
  financial fraud detection be organized into a clear taxonomy, and what
  benchmark datasets are used to evaluate them?
- *RQ2 (Results):* How do different GNN models compare in practice on public
  benchmarks—including against tabular baselines—and what gaps remain in
  current evaluation protocols?
- *RQ3 (Discussion — Beyond GNNs):* What are the main limits of current GNNs,
  and what next-generation approaches can go beyond them to improve real-world
  fraud detection?

The remainder of this paper is organized as follows. Section II presents the related work and existing taxonomies on graph-based financial crime detection. Section III details the methodology and dataset selection criteria. Section IV reviews the benchmark datasets and evaluation protocols, while Section V analyzes the core Graph Neural Network architectures and empirical results. Finally, Section VI discusses open challenges, deployment constraints, and future research directions.



= Methodology <sec:method>
// TODO: research questions, databases searched, search query, inclusion/exclusion??
// calude suggestion: criteria, PRISMA-style selection flow (records found -> screened -> included).


== Use of Generative AI Tools <sec:ai_use>

Here we disclose the use of generative AI tools during the preparation of this review (@tab:ai_use). These tools
supported our work but did not replace it: the authors still selected,
analysed, and checked AI outputs against the original papers. We
take full responsibility for the content of this paper.

#figure(
  caption: [Use of generative AI tools during the preparation of this review.],
  placement: none,
  {
  set par(justify: false)
  table(
    columns: (1fr, 0.9fr, 1.4fr),
    align: left,
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Task][AI Tool(s)][Role of the Authors],
    [*Search and validation of sources*], [NotebookLM (Google), Perplexity], [Final inclusion decisions; each source checked in its original publication],
    [*Synthesis and summary of sources*], [NotebookLM, Perplexity, Claude (Anthropic)], [Authors also read and summarized the papers themselves, then compared their own understanding with the different AI syntheses],
    [*Retrieval of key facts and figures*], [Claude], [Figures checked against the cited paper],
    [*Text editing and proofreading*], [Claude], [Arguments written by the authors; edits reviewed before acceptance],
  )
  },
) <tab:ai_use>
== Background <sec:background>

=== Graph representation and message passing

Financial transactions can be represented as a graph
$cal(G) = (cal(V), cal(E))$. The nodes $cal(V)$ represent entities such as
bank accounts, credit cards, merchants, or devices. The edges $cal(E)$
represent relationships between them. For example, a money transfer is a
directed edge from the sending account to the receiving account. Two accounts
that use the same device can each be linked to the node representing that
device.

A financial graph is heterogeneous when it contains multiple types of nodes
or edges. It is attributed when these nodes or edges carry additional
information. Each node $v$ can have a feature vector $bold(x)_v$, containing
information such as an account's age or balance. Each edge $(u, v)$ can have
a feature vector $bold(x)_(u v)$, containing information such as the amount
and timestamp of a transfer.

A graph neural network (GNN) uses *message passing* to combine information
about an entity with information from its neighbors. For example, the
activity of the accounts that send money to an account can provide additional
context for assessing its fraud risk.

The initial representation of node $v$ is $bold(h)_v^((0)) = bold(x)_v$.
At layer $k$, the node receives messages from its incoming neighbors
$cal(N)(v)$ and aggregates them as follows.

$
  bold(m)_v^((k)) =
    plus.o_(u in cal(N)(v))
    "MSG"^((k))(
      bold(h)_u^((k-1)),
      bold(h)_v^((k-1)),
      bold(x)_(u v)
    )
$

The function $"MSG"$ computes the message from neighbor $u$ using the
previous representations of the two nodes and, where available, the edge
features. The operator $plus.o$ combines the messages independently of their
order, for example by taking their sum or mean. The update function then
combines the aggregated message with the previous representation of $v$.

$
  bold(h)_v^((k)) =
    "UPDATE"^((k))(bold(h)_v^((k-1)), bold(m)_v^((k)))
$

After $K$ layers, a node can incorporate information from entities up to $K$
steps away along the paths followed by the messages. This can help identify
indirect connections in money laundering chains. The resulting node
representations can be used to assign a fraud risk score to an account or
transaction.

#figure(
  block(
    width: 100%,
    fill: rgb("#f2f6fa"),
    stroke: 0.6pt + rgb("#9eb5c8"),
    radius: 4pt,
    inset: 10pt,
  )[
    *Raw transactions* $arrow.r$ *Financial graph*
    $arrow.r$ *Message passing* $arrow.r$ *Fraud risk score*
  ],
  caption: [From raw transactions to a graph-based fraud risk score.],
) <fig:mpnn_pipeline>

When the target is a group of accounts, the model needs a representation of
the candidate subgraph. On the Elliptic2 benchmark, simply pooling the node
representations produced results close to random. Classification improved
when the model was told which nodes belonged to the candidate subgraph and
could use information from the surrounding graph (Bellei et al., 2024).

=== GNN architectures for financial fraud detection

Motie and Raahemi (2024) group GNN models for financial fraud detection into
four families according to how they gather and combine information from the
graph.

#figure(
  image("figures/taxonomy_gnn.jpg", width: 100%),
  caption: [Taxonomy of GNN models for financial fraud detection.],
) <fig:taxonomy>

*Convolutional models.* Graph Convolutional Networks (GCNs) aggregate neighbor
features using weights based on node degrees. They provide a baseline for
static transaction graphs. GraphSAGE typically samples a fixed number of
neighbors and combines their features with those of the target node. Its
learned aggregation function can be applied to new accounts when their
features and connections are available.

*Attention-based models.* Graph Attention Networks (GATs) learn a weight
$alpha_(u v)$ for the contribution of neighbor $u$ to node $v$. This lets the
model give more weight to the neighbors that are most informative for its
prediction.

*Meta-path-based models.* These models aggregate information along predefined
sequences of node or edge types in a heterogeneous graph. For example, an
account–device–account meta-path connects accounts that use the same device.

*Transformer-based models.* These models use self-attention to combine
information across the graph. Depending on the architecture, attention can
incorporate information from beyond the direct neighbors of a node.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    inset: 5pt,
    stroke: 0.4pt + rgb("#ccd6df"),
    table.header(
      [*Model*], [*Core idea*], [*Advantage*], [*Limitation*],
    ),
    [*GCN*],
    [Degree-normalized aggregation of neighbor features.],
    [Simple baseline for static transaction graphs.],
    [The original transductive setup assumes that target nodes are already
      present in the training graph.],
    [*GraphSAGE*],
    [Samples neighbors and combines their features with those of the target
      node.],
    [Can score unseen accounts using their features and connections.],
    [Sampling can miss rare but informative connections.],
    [*GAT*],
    [Learns attention weights for neighboring nodes.],
    [Can give more weight to informative connections.],
    [Attention over many edges can increase memory use and latency.],
  ),
  caption: [Comparison of three foundational GNN architectures.],
) <tab:gnn_backbones>

We extend the taxonomy of Motie and Raahemi (2024) with the models covered in
our results. ATGNN-SP uses a simplified two-hop graph convolution (Yue et
al., 2026). Three other studies use GraphSAGE (Silva et al., 2026,
Sumsuzzaman et al., 2026, and Dintakurthy et al., 2026). LAS-GNN, MEGA-PNA,
and GLASS form an additional message-passing branch. They use ego-ID message
passing, Principal Neighbourhood Aggregation (PNA), and subgraph labeling,
respectively (Verlaan et al., 2025, Bilgi et al., 2026, and Bellei et al.,
2024).

C2GAT and HTGNN belong to the attention family. Both use graph attention, and
HTGNN also uses transformer-style attention to combine entity representations
(Chen et al., 2026, and Nguyen et al., 2025). The meta-path and Transformer
branches retain the models from the original taxonomy.

// To include the blue diagram, put the PNG at figures/taxonomy-blue.png
// relative to this .typ file, then remove the opening and closing comment
// markers around the figure below.
/*
#figure(
  image("figures/taxonomy-blue.png", width: 100%),
  caption: [
    Taxonomy of GNN models for financial fraud detection, adapted from
    Motie and Raahemi (2024) and extended with the models covered in our results.
  ],
) <fig:taxonomy>
*/

=== Training paradigms and label availability

Confirmed fraud cases are rare, and investigations can delay labeling.
*Supervised learning* uses verified historical labels. Weighted or focal
losses can help account for the imbalance between fraudulent and legitimate
examples.

In *semi-supervised learning*, only some accounts or transactions are
labeled. The model uses those labels for training while drawing on the
features and connections of unlabeled entities through message passing.

When confirmed fraud labels are unavailable, *unsupervised learning* can
search for unusual patterns through graph reconstruction or subgraph mining.
*Self-supervised learning* derives training signals from the graph itself.
For example, contrastive learning can train a model to bring the
representations of two views of the same entity or subgraph closer together.
The learned representations can then be used to identify unusual patterns.

#figure(
  table(
    columns: (auto, 1fr, 1fr, auto),
    align: left,
    inset: 5pt,
    stroke: 0.4pt + rgb("#ccd6df"),
    table.header(
      [*Paradigm*], [*Label availability*], [*How it learns*], [*Example*],
    ),
    [*Supervised*],
    [Verified historical labels are available.],
    [Minimizes a classification loss on labeled examples.],
    [PC-GNN],
    [*Semi-supervised*],
    [Only a subset of entities or transactions is labeled.],
    [Uses labeled examples for training and draws on the surrounding graph.],
    [GTAN],
    [*Unsupervised*],
    [Confirmed fraud labels are unavailable.],
    [Searches for unusual patterns in the graph structure or attributes.],
    [Subgraph mining],
    [*Self-supervised*],
    [Training signals are derived from the graph.],
    [Learns from a task such as aligning two views of the same subgraph.],
    [Audit-HCL],
  ),
  caption: [Comparison of training paradigms by label availability.],
) <tab:learning_paradigms>

== Existing review papers and limits
// TODO A faire par aymen, bientot fait

Multiple research papers with similar research questions have been published in the last 3 years @motie_financial_2024 @cheng_graph_2025 @deprez_advances_2025. In this section we synthesize their findings to establish a foundation for new developments in the subject of fraud detection. Recurring themes and convergences across the reviews are summarized in @tab:convergences.

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
    [Reference dataset], [Elliptic], [Elliptic], [Elliptic, IBM AML],
  )
) <tab:convergences>

All three reviews agree that collective fraud detection (groups, communities, subgraphs) remains a gap. GNN-based financial fraud detection is overwhelmingly framed as node-level and, occasionally, edge-level detection. Motie and Raahemi @motie_financial_2024 state explicitly that no reviewed paper covered graph-level anomaly detection or the detection of abnormal communities, although such anomalies are typical in finance.

Deprez et al. @deprez_advances_2025 operationalize money laundering typologies as labels (the 8 patterns of the IBM AML dataset: fan-in, fan-out, bipartite, stack, cycle, random, scatter-gather, gather-scatter). However, each transaction is still classified individually with its pattern label, and the model never outputs a ring or a pattern as an object. In 2025, multi-node pattern detection is therefore still approached through an edge-level proxy, not through genuine subgraph detection.



== Technical foundations of Graph Neural Networks

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
  placement: none,
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

Across these scenarios, four obstacles keep coming back. Fraudsters imitate
normal behavior to escape detection @dou_enhancing_2020, fraud examples are
extremely rare compared with normal ones @liu_pick_2021, criminals constantly
adapt their tactics, which forces models to keep learning over time
@deprez_advances_2025, and banks need fast decisions together with
explanations that regulators can audit @rao_xfraud_2021
@khanvilkar_regulatory_2025. In collusion rings, confirmed labels may even
arrive months after the facts @jiao_dynamic_2026. These obstacles motivate the
technical choices described in the next section.

#figure(
  caption: [Main financial fraud scenarios studied with graph learning.],
  placement: none,
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


== Taxonomy of GNN-Based Approaches <sec:taxonomy>
// Following @motie2024financial, existing models can be grouped into four families
// (@fig:taxonomy): convolutional-based, attention-based, metapath-based and
// transformer-based models.

// #figure(
//   image("figures/motie2024-taxonomy.jpg", width: 100%),
//   caption: [Taxonomy of GNN models for financial fraud detection, from @motie2024financial.],
// ) <fig:taxonomy>


= Datasets <sec:datasets>
// Draft (2026-11-15).

Public datasets for graph-based fraud detection differ in their domain, in
the level at which labels are given (node, edge, or subgraph), and in whether
the data are real or simulated. @tab:datasets summarizes the datasets used by
the studies reviewed in this paper.

#figure(
  caption: [Public datasets used for graph-based financial fraud detection.],
  placement: top,
  scope: "parent",
  {
  set par(justify: false)
  set text(size: 8.5pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto, 3.4cm),
    align: left,
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Dataset][Domain][Nodes / Edges][Label level][Positives][Data][Used in],
    [*Elliptic* @weber_anti-money_2019], [Bitcoin], [204K / 234K], [Node (transaction)], [2% illicit], [Real], [@yue_adaptive_2026 @kulkarni_dynberg_2025 @deprez_advances_2025],
    [*Elliptic2* @bellei_shape_2024], [Bitcoin], [49.3M / 196.2M], [Subgraph], [2.3% of 122K subgraphs], [Real, heuristic labels], [@bellei_shape_2024],
    [*ETH phishing* @wu_who_2022], [Ethereum], [3.0M / 13.6M], [Node (address)], [1,165 addresses], [Real], [@egressy_provably_2024 @bilgi_mega_2026],
    [*IBM AML* @altman_realistic_2023], [Bank transfers], [0.5–2.1M / 5–180M], [Edge + 8 patterns], [0.05–0.12%], [Synthetic], [@deprez_advances_2025 @egressy_provably_2024 @bilgi_mega_2026 @sumsuzzaman_graph_2026],
    [*AMLSim* @suzumura_amlsim_2021], [Bank transfers], [Configurable], [Edge + patterns], [Configurable], [Synthetic], [@silva_efficient_2026],
    [*T-Finance* @tang_rethinking_nodate], [Transactions], [39K / 21.2M], [Node (account)], [4.58%], [Real], [@tang_rethinking_nodate],
    [*DGraph* @huang_dgraph_2022], [Consumer loans], [3.7M / 4.3M], [Node (user)], [1.3% of labeled users], [Real], [@huang_dgraph_2022],
    [*FinFraudBench* @chen_finfraudbench_2026], [Card / bank payments], [up to 9.0M / 89.2M], [Node (transaction)], [0.15–0.52%], [Kaggle tables], [@chen_finfraudbench_2026],
    [*IEEE-CIS* @ieee-cis_fraud_2019], [E-commerce], [Tabular (590K tx)], [Transaction], [3.5%], [Real], [@sha_detecting_2025 @nguyen_real-time_2025],
    [*PaySim* @lopez-rojas_paysim_2016], [Mobile money], [Tabular (6.4M tx)], [Transaction], [0.13%], [Synthetic], [@nguyen_real-time_2025],
    [*BankSim* @lopez-rojas_banksim_2014], [Card payments], [Tabular (595K tx)], [Transaction], [1.2%], [Synthetic], [@dintakurthy_multisource_2026],
  )
  },
) <tab:datasets>

*Cryptocurrency datasets.* Because blockchain data are public,
cryptocurrencies provide most of the real labeled transaction graphs. The
Elliptic dataset @weber_anti-money_2019 is a Bitcoin graph of 203,769
transactions (nodes) linked by 234,355 directed payment flows (edges), with
166 node features and 49 time steps. About 2% of the transactions are labeled
illicit and 21% licit; the others are unknown. It is the most common benchmark
in the GNN fraud literature @motie_financial_2024, but it only supports
node-level classification.

Elliptic2 @bellei_shape_2024 @elliptic_elliptic2_2024 is, to our knowledge,
the first large labeled benchmark for subgraph classification in finance. Each
node is a cluster of Bitcoin addresses believed to be controlled by the same
entity, and each edge is a transaction between two clusters. The background
graph covers one year of blockchain activity, with 49.3M nodes, 196.2M edges,
43 node features and 95 edge features. Inside it, 121,810 subgraphs are
labeled: 2,763 (2.3%) are suspicious and the others licit. A subgraph is
suspicious when it connects an illicit cluster to a licit one, which suggests
an attempt to convert criminal proceeds into fiat money or other
cryptocurrencies @elliptic_elliptic2_2024. These labels come from a path
heuristic (at most six hops) and therefore indicate suspicion, not confirmed
laundering @bellei_shape_2024. The intended task is binary subgraph
classification, and the data are released on Kaggle under a non-commercial
licence (CC BY-NC-ND 4.0).

On Ethereum, the phishing network collected by Wu et al. @wu_who_2022
contains 2,973,489 addresses and 13,551,303 transactions with amounts and
timestamps, but only 1,165 addresses are labeled as phishing, from reports on
Etherscan. It serves as the real-data test for multigraph GNNs
@egressy_provably_2024 @bilgi_mega_2026.

*Synthetic AML datasets.* Real bank transfers are usually confidential, so
pattern-level AML research relies on simulators. AMLSim
@suzumura_amlsim_2021 is an open-source multi-agent simulator from IBM that
injects known laundering patterns into synthetic bank transactions. It was
used, for example, to generate graphs of 100K and 1M accounts
@silva_efficient_2026. The IBM AML datasets @altman_realistic_2023 come from a
more realistic generator: it models individuals, companies, and banks, and
follows illicit funds through any number of transfers, so that every
transaction can be labeled. Six datasets are released, in two groups with a
higher (HI) or lower (LI) laundering rate and three sizes each, from 515K
accounts and 5M transactions (HI-Small) to about 2.1M accounts and 180M
transactions (HI-Large). Only one transaction in 807 to 1,948 is laundering.
Part of the laundering follows eight patterns (fan-in, fan-out, bipartite,
stack, cycle, random, scatter-gather, and gather-scatter), listed in a
separate file. IBM AML is therefore the reference dataset for pattern-aware
models @egressy_provably_2024 @deprez_advances_2025 @bilgi_mega_2026.
However, its labels are generated by the same patterns that these models are
designed to find, so results may not transfer to real data.

*Real fraud graphs with node labels.* T-Finance @tang_rethinking_nodate links
39,357 anonymized accounts by 21,222,543 transaction edges, with 10 features
per account (e.g., registration days and login activity). Experts labeled
4.58% of the accounts as anomalous, for fraud, money laundering, or online
gambling. DGraph @huang_dgraph_2022 comes from Finvolution, a Chinese consumer
lending platform. Its 3,700,550 nodes are users and its 4,300,999 directed,
timestamped edges are emergency-contact links. Only 1,225,601 users are
labeled, of which 15,509 (1.3%) are fraudsters; the remaining 2.5M nodes are
unlabeled background users kept to preserve the structure. Both graphs are
homogeneous. FinFraudBench @chen_finfraudbench_2026 keeps several entity
types instead. Its two graphs, CreditCard-Fraud and BankTrans-Fraud, are built
from public Kaggle transaction tables and contain six node types
(transaction, customer, card, merchant, category, and location) and fourteen
directed edge types, with up to 8.99M nodes, 89.23M edges, and natural fraud
rates of 0.52% and 0.15%. Note that the credit-card table is itself simulated
(Sparkov generator). YelpChi and Amazon, two review-fraud graphs, are also
frequent in graph anomaly detection @dou_enhancing_2020 @liu_pick_2021, but
they are not financial and are not detailed here.

*Tabular datasets turned into graphs.* Several studies build their own graph
from tabular transaction logs. IEEE-CIS @ieee-cis_fraud_2019 contains real
e-commerce transactions provided by Vesta (about 590K labeled, 3.5% fraud),
whose card, address, e-mail, and device fields are used to link transactions
@sha_detecting_2025 @nguyen_real-time_2025. PaySim
@lopez-rojas_paysim_2016 simulates one month of mobile-money transactions,
calibrated on real logs from an African country (6.4M transactions in its
Kaggle version, 0.13% fraud). BankSim @lopez-rojas_banksim_2014 simulates six
months of card payments, calibrated on data from a Spanish bank (594,643
payments, of which 7,200 are fraudulent) @dintakurthy_multisource_2026.
Because each paper chooses its own nodes and edges, results on these datasets
are hard to compare.

Two general observations can be made. First, no public dataset combines real bank
transfers with group-level labels: the only subgraph labels (Elliptic2) come
from a heuristic on Bitcoin data, and the only typology labels (IBM AML,
AMLSim) are synthetic. Studies on real bank data use private graphs
@johannessen_finding_2023. Second, all datasets are highly imbalanced, with
0.05% to 4.6% positives. Consequently recent work reports PR-AUC or
minority-class F1 rather than accuracy as a metric.

= Results

// Tableau de resultats: Shehrazade (draft table inserted 2026-11-16)

In this section we synthesize the results of 9 studies not covered by the recent review papers, all of them released since 2024. @tab:results lists the main results reported by each study.

#figure(
  caption: [Main results reported by recent studies (2024–2026). Values cannot be compared across rows because datasets, tasks and metrics differ. "Best baseline" is the strongest competing model reported by the authors; "private" means non-public industrial data.],
  placement: top,
  scope: "parent",
  {
  set par(justify: false)
  set text(size: 8.5pt)
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: left,
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    table.header[Study][Dataset][Task][Metric][Proposed model][Best baseline],
    [Bellei et al. @bellei_shape_2024], [Elliptic2], [Subgraph classification], [PR-AUC / ROC-AUC], [GLASS 0.208 / 0.889], [GNN-Seg 0.026 / 0.537; Sub2Vec 0.022 / 0.496],
    [Verlaan et al. @verlaan_-gnn_2025], [Watts–Strogatz (synthetic)], [Temporal motif key nodes], [AP (GS / SG / C3)], [LAS-GNN 1.00 / 0.97 / 1.00], [GIN+Ego+HMP 0.86 / 0.86 / 0.86],
    [Verlaan et al. @verlaan_-gnn_2025], [Watts–Strogatz (synthetic, balanced)], [Temporal 6-cycle key nodes], [F1], [LAS-GNN 0.87], [GIN+Ego+HMP 0.49],
    [Bilgi and Atasu @bilgi_mega_2026], [IBM AML Medium-HI], [Edge classification], [Minority F1 (%)], [MEGA-PNA 78.26], [Multi-PNA 76.12],
    [Bilgi and Atasu @bilgi_mega_2026], [IBM AML Large-LI], [Edge classification], [Minority F1 (%)], [MEGA-PNA 38.31], [Multi-FraudGT 35.49],
    [Bilgi and Atasu @bilgi_mega_2026], [ETH phishing], [Node classification], [Minority F1 (%)], [MEGA-PNA 64.84], [Multi-PNA 64.61; DIAM 64.43],
    [Silva and Lima de Castro @silva_efficient_2026], [AMLSim 100K accounts], [Edge ranking], [AP / Recall\@1%], [GraphSAGE 0.783 / 0.877], [LightGBM 0.967 / 0.990],
    [Silva and Lima de Castro @silva_efficient_2026], [AMLSim 1M accounts], [Edge ranking], [Recall\@1%], [GraphSAGE 0.870], [—],
    [Nguyen and Le @nguyen_real-time_2025], [IEEE-CIS], [Transaction classification], [AP / ROC-AUC], [HTGNN 0.641 / 0.940], [BRIGHT 0.371 / 0.863; LightGBM 0.361 / 0.851],
    [Nguyen and Le @nguyen_real-time_2025], [PaySim], [Transaction classification], [AP / ROC-AUC], [HTGNN 0.682 / 0.956], [BRIGHT 0.475 / 0.891],
    [Yue et al. @yue_adaptive_2026], [Elliptic (steps 35–49)], [Node classification], [AP / F1], [ATGNN-SP 0.262 / 0.265], [Graph HGB 0.262 / 0.271],
    [Chen and Yang @chen_real-time_2026], [Taobao (private)], [Transaction classification], [P\@20%R (%)], [C2GAT framework 86.2], [MLP + engineered features 68.5],
    [Chen and Yang @chen_real-time_2026], [Offline merchants (private)], [Transaction classification], [P\@20%R (%)], [C2GAT framework 56.5], [MLP + engineered features 47.2],
    [Dintakurthy @dintakurthy_multisource_2026], [BankSim (10K-row extract)], [Transaction classification], [PR-AUC], [GraphSAGE + XGBoost fusion 0.450 (GraphSAGE alone 0.075)], [XGBoost 0.425],
    [Sumsuzzaman et al. @sumsuzzaman_graph_2026], [IBM AML HI-Small], [Edge classification], [AUPRC], [Edge GraphSAGE 0.166], [MLP 0.210],
  )
  },
) <tab:results>

== From node scores to fraud structures

The main advance since the existing reviews is a new emphasis on subgraphs for AML detection. New models and data where laundering is treated as a structure rather than as a set of independent transactions have emerged. Notably, experiments on Elliptic2 show that a subgraph method that marks subgraph membership and uses the surrounding graph (GLASS) outperforms pooling node embeddings (PR-AUC 0.208 vs 0.026) @bellei_shape_2024.

At the motif level, LAS-GNN extends ego-ID message passing to temporal motifs. It also detects the key accounts of cycles and scatter-gather schemes in synthetic networks @verlaan_-gnn_2025. Another model, MEGA, preserves repeated transfers between the same pair of accounts and improves laundering detection on all IBM AML variants @bilgi_mega_2026. However, on bank-transfer data, these models still output a score per node or per transaction. Groups have to be rebuilt afterwards, at the cost of rapidly decreasing purity @silva_efficient_2026.


== Comparison with tabular models

Four studies compared GNNs with gradient-boosted trees trained on the same features and split chronologically. In all four, the best GNN alone was at best on par with the tree model (@tab:results). Two studies report large GNN gains, but compare them with a model that received less information @nguyen_real-time_2025 or only with a multilayer perceptron @chen_real-time_2026.

= Discussion and conclusion

Among the 9 studies, no two share the same dataset, task and metric. This results in an apples-to-oranges comparison, and no method can be shown to be better than another across studies. In turn, this makes it difficult to confidently draw general conclusions about the state of the art of this domain. This at least answers our second RQ: gaps still remain in evaluation protocols. Additionally, the gap is wider for bank transfers: real data remain confidential, so studies rely on synthetic data.

One development to be noted is how studies deal with imbalanced datasets. Recent studies no longer use accuracy. Instead, they use the area under the precision-recall curve (PR-AUC), the F1-score of the fraud class, and the share of frauds found among the top-ranked alerts (precision or recall at k). Some also define their own custom metrics.

Two gaps are partly closed: heterogeneous graphs and subgraph-level detection. However, once again, most of this progress (and in fact all of the subgraph progress) was obtained on crypto or synthetic datasets.

The clearest progress concerns time: whereas half of the studies covered by Motie and Raahemi used static graphs @motie_financial_2024, recent models treat time as the order of transactions inside a pattern @verlaan_-gnn_2025, as timestamps encoded in attention or recurrent layers @chen_real-time_2026 @nguyen_real-time_2025, or as a stream on which the model is updated @yue_adaptive_2026, and they split data chronologically.

One of the review papers @cheng_graph_2025 states that GNNs significantly outperform traditional machine learning methods. But when recent studies compare GNNs with gradient boosting under a chronological split, the GNN alone is not better @silva_efficient_2026 @yue_adaptive_2026 @dintakurthy_multisource_2026 @sumsuzzaman_graph_2026. These results do not mean that graphs are useless, but that well-tuned tree models can capture most of the signal and the graph does not add much when used alone. A fair evaluation of GNNs for fraud therefore needs a gradient-boosting baseline trained on the same features and the same chronological split.


== Open Challenges and Future Directions <sec:challenges>
// TODO: edge- and graph-level anomaly detection; unsupervised and semi-supervised
// learning; dataset scarcity (taxation, real estate, mortgage); dynamic graphs.

A common thread across the recent studies is the reliance on cryptocurrency or synthetic datasets. It was already the case in the studies covered by the existing reviews (@tab:convergences), and it remains so today. There is no public benchmark of real bank transfers with group-level labels (see @sec:datasets): Elliptic2 is crypto, IBM AML and AMLSim are synthetic. Banking secrecy, competition and data protection laws will likely keep this issue open in the near future.

Unsupervised or label-free collective detection remains understudied. None of the 9 studies of our corpus covers it, although Motie and Raahemi @motie_financial_2024 called for more unsupervised methods. A recent exception outside our corpus is Audit-HCL @jiao_dynamic_2026, which reports useful detection even without fraud labels.

= Conclusion

This review addressed three research questions on graph neural networks for detecting financial fraud in transaction networks. For the first, we organized the GNN architectures used in finance into a taxonomy and described the benchmarks and datasets on which they are evaluated. Most of these datasets are either synthetic or built from cryptocurrency transactions; no public dataset combines real bank transfers with group-level labels.

For the second, we found that results are hard to compare: no two of the recent studies share the same dataset, task and metric, so no model can be declared better than another across studies. Within studies, well-tuned tree-based models such as XGBoost remain competitive with GNNs, which are most useful when fraud is defined by a structure, such as a cycle or a laundering path, or when they are combined with a tabular model.

For the third, we observed clear progress over the last two years, notably in the modeling of time and in the evaluation of highly imbalanced data. Important gaps remain, however: group-level detection is still rare outside cryptocurrency and synthetic data, unsupervised methods are scarce, and few studies release their code. A common evaluation protocol, with public data, chronological splits and a strong tabular baseline, would be a first step toward comparable results.

