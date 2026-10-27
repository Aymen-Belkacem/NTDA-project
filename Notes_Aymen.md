# Notes personnelles

## Review paper qui fait une bonne partie de notre travail

Le titre est "Financial fraud detection using graph neural networks: A systematic review" (10.1016/j.eswa.2023.122156). 

La conclusion indique que plusieurs qu'il y a un besoinde plus d'exploration pour "dynamic and heterogeneous graphs, particularly in cross-domain applications". "Moreover, we highlighted the limited availability of diverse graph datasets in some domains, such as taxation, and noted that certain areas of finance, such as real estate and mortgage, to name a few, have not been thoroughly explored."

"Although the majority of the papers we reviewed employed supervised methods for fraud detection, we advocate for a more extensive investigation into unsupervised and semi-supervised approaches. These methods are more practical and closer to real-world scenarios, as they hold the potential to reveal previously unseen patterns."

Résumé du review paper:

- 5 bases de données académiques et 33 articles choisis pour analyser l'etat de l'art actuel.
- Taxonomy: 4 catégories: Convolutional-based, Attention-based, Metapath-based, and Transformer-based models
- Principalement apprentissage supervisés
- Domaines d'application: cryptocurrency, credit cards, online payments, taxation, and insurance
- Types de graphes: homogeneous, heterogeneous, static, temporal, and dynamic graphs
- Lacunes de recherche:  Limited edge-level and graph-level anomaly detection; need for more unsupervised approaches

!["test"](./Images/1-s2.0-S0957417423026581-gr4_lrg.jpg)


## Review paper de 2024

Titre: "Graph Neural Networks for Financial Fraud Detection: A Review".

### Abstract

- This review reveals that GNNs are exceptionally adept at capturing complex relational patterns and dynamics within financial networks, significantly outperforming traditional fraud detection methods.
- This review not only highlights the potential of GNNs to improve fraud detection mechanisms but also identifies current gaps and outlines future research directions
to enhance their deployment in financial systems.
- 100 studies reviewed

### 1 Introduction

- Traditional rule-based systems and classic machine learning methods have been used for fraud detection, but they often struggle with complex fraud patterns on large amounts of financial data [5–11].
- Despite the potential of GNNs in fraud detection, there are practical challenges that need to be addressed. These include managing large-scale financial graph data and ensuring model adaptability to dynamic patterns [33–37].
- this study proposes the following detailed research questions (RQs):
    - RQ1: How can various GNN methodologies in financial fraud detection be understood through a unified framework?
    - RQ2: Why use GNNs for financial fraud detection? What roles do they play?
    - RQ3: How to design GNNs suitable for financial fraud detection?
    - RQ4: How are GNNs deployed in real-world financial fraud detection scenarios, and what are their impacts?
    - RQ5: What are the current challenges and futuredirections for GNNs in financial fraud detection?
- Unified framework for categorizing and analyzing GNN methodoloies. Implementation here: https://github.com/AI4Risk/antifraud. See also https://github.com/AI4Risk/awesome-fraud-detection#-graph-based-fraud-detection.

### 2 Preliminaries

- Homogeneous and Heterogeneous Graphs. A graph G =  (V, E) can be extended to a heterogeneous graph defined as  G = (V, E, φ, ψ), where φ : V → A assigns types to nodes  and ψ : E → R assigns types to edges. A graph is homogeneous if φ and ψ each map to a single type; otherwise, it is heterogeneous.
- Typically, GNNs employ a message passing mechanism that incrementally aggregates information from the neighborhoods.
- 3 levels of tasks for GNNs: node classification, edge classification, and graph classification + attribute prediction.

### 3 Graph Neural Networks in Financial Fraud  Detection

- This section categorizes the GNNs in financial fraud detection into four main types
    - Graph Convolutional Networks
        - Graph Convolutional Networks (GCNs) process data structured in graph form, effectively identifying cross-transaction fraudulent activity patterns within transaction networks.
        - GCNs efficiently learn features and recognize patterns in financial data by capturing local connectivity and automatically discerning node and edge features. However, GCNs can suffer from over-smoothing, leading to the loss of distinct node characteristics, and are prone to overfitting with limited data.
    - Graph Attention Networks
        -Graph Attention Networks (GATs) employ an attention mechanism, allowing the model to focus on the most relevant parts of the graph.
        - The attention mechanism is particularly useful in applications such as financial fraud detection [55]. GATs outperform GCNs by dynamically allocating attention weights to neighbors, enabling adaptive focus on relevant nodes for improved complex pattern detection in financial fraud.
    - Graph Temporal Networks
        - Graph Temporal Networks (GTNs) incorporate the dynamic changes in financial transactions, utilizing expressions like:  H(t+1) = σ( A(t) H(t)W(t)) (8)  to capture dynamic transactional data.
        - These networks are particularly effective in identifying fraud within datasets sensitive to temporal patterns, such as credit card transactions  and high-frequency trading [56, 57].
        - To enhance the model’s sensitivity to temporal dynamics, a temporal attention mechanism can be introduced, enabling the model to focus on the most significant parts of the graph at different times.
    - Heterogeneous Graph Neural Networks
        - Heterogeneous Graph Neural Networks (HGNNs) are designed to handle graphs composed of different types of nodes and edges, enabling the model to capture a richer set of relationships and properties within the data.
        - Moreover, incorporating an attention mechanism [62–64] can dynamically weight the importance of different types of relationships:
- GCNs such as Skip-GCN [65] and EWS-GCN [66] are  adept at leveraging node and edge features but may fall short in capturing the temporal dynamics of fraud patterns.
- GATs like ASA-GNN [67] and ABGRL [68], offer adaptive attention mechanisms that enhance the detection of nuanced fraudulent behaviors, yet they can be computationally intensive.
- GTNs such as STAGN [20] and STGN [69], proficient in  detecting time-sensitive fraud patterns, require more computational overhead due to their temporal attention modules.
- HGNNs, including LIFE [70] and MultiFraud [71], excel at  integrating heterogeneous information, providing a rich context for fraud detection, but their complexity can be a double-edged sword, demanding more resources and making them less suitable for scalability-focused applications.
- Despite their efficacy, GNNs encounter significant scalability challenges when scaling up to large financial networks. The high memory and computational demands on a single machine primarily stem from GPU memory constraints and the extensive time required for matrix multiplication operations. Additionally, in distributed environments, the high communication costs related to obtaining and combining graph embeddings of nodes’ multi-hop neighbors in each training batch hinder the speed benefits of parallelization [12]. These challenges highlight the need for more efficient GNN architectures to handle larger financial networks with quick response times for real-time fraud detection.

### 4 Reasons to Choose GNNs for Financial Fraud  Detection

- Transaction Graph Construction. Transaction data is pivotal for revealing fraudulent activities. Creating a transaction network (graph), with nodes representing individual accounts or entities (such as individuals, companies) and edges representing transactions (e.g., transfers, payments) [74, 75], enables the effective identification of abnormal transaction patterns [76].
- Relationship-Based Graph Construction. Constructing complex network (graph) that include various entities (like individuals, companies, bank accounts) and their relationships (such as ownership, control relations) can reveal the cooperation networks behind complex fraudulent activities like money laundering or insider trading [79].
- Behavioral and Device Pattern Graph. Constructing graphs  based on user behaviors and device usage patterns (e.g., logins, queries, transaction requests, device types, operating systems, and device IDs) can reveal the similarities or abnormal patterns between different accounts [81–83].
- Information Flow Graph Construction. Constructing information flow graphs [84] by analyzing the relationship between public information (e.g., news, announcements, social media discussions) and financial market activities can help identify patterns of information manipulation and the centers of influence, thereby detecting related fraudulent activities [85–87].
- The foundational layer of GNNs, the message passing mechanism, enables a sophisticated form of feature extraction. It integrates attributes from both nodes and edges, which are rich in data that can signal fraudulent activity. This mechanism effectively combines the influences of neighboring nodes and updates each node’s features through successive layers.
- The capacity for automatic feature learning—requiring no specific programming for feature identification—makes GNNs exceptionally effective at uncovering both sophisticated and previously undetected fraud schemes. Overall, the intrinsic feature learning of GNNs not only eases model development and training but also ensures rapid adaptation to new fraudulent tactics, play ing a vital role in fighting financial crime.
- Challenges Posed by Dynamic Financial Graphs. The everchanging structure of financial graphs complicates the task of fraud detection. As new nodes and edges are added, previously learned patterns may no longer apply, and new patterns of fraud can emerge.
    - Temporal Attention Mechanisms
    - Time-Decay Functions
    - Graph Temporal Networks
    - Dynamic Embeddings.
- Multimodal Data Fusion. GNNs emerge as a powerful tool  for fusing different types of data within a unified framework. GNNs can naturally accommodate multimodal data by representing them as nodes in a graph, where the edges capture the relationships between various data points [92].

### 5 GNN Design for Financial Fraud Detection

- Architectural choices
    - Graph Convolutional Networks excel in structured data environments, capturing local transaction patterns effectively.
    - Graph Attention Networks leverage attention mechanisms to emphasize important transactions or patterns, beneficial in scenarios with significant edge attributes. 
    - Graph Transformer Networks adeptly handle time-series data, tracking financial activities over periods.
    - Heterogeneous Graph Neural Networks are designed for networks with diverse node and edge types, enabling comprehensive integration of various financial behaviors.
- Feature selection
    - Node Feature Engineering Node feature engineering in financial contexts is crucial. It involves a strategic analysis and extraction of node attributes, extending beyond basic transaction details like amount, frequency, and timing [94].
    - Edge Features Integration
    - Multimodal Data
        - Self-Attention Weighted Fusion.
        - Multimodal Embedding Fusion Layers.
        - Graph Convolution Interactions.
    - Imbalanced Data
        - Graph Data Augmentation.
        - Ensemble Methods in Graph Learning.
        - Cost-sensitive GNN.
        - Adaptive Resampling in GNN.
        - Meta Paths in Heterogeneous Graph Networks.
        - Anti-Money Laundering Detection
### 6 Application in Financial Fraud Detection

- Credit Card Fraud Detection
- Online Payment Fraud Detection
- Insurance Fraud Detection 

### 7 Future Directions

- Integration of Reinforcement Learning with GNNs.
- Application of Adversarial Learning in GNNs.
- Pre-training in GNNs.
- Enhancing the Explainability of GNNs.
- Enhancing the Scalability of GNNs.

### 8 Conclusion

- The introduction of a unified framework helps to systematize the diverse approaches observed in existing literature, offering a clearer understanding of how GNNs can be effectively applied in complex financial contexts.

## Datasets


- **Elliptic (Bitcoin Payment Flow):** A widely cited temporal graph mapping Bitcoin transactions[1]. It contains **203,769 address nodes** and **234,355 transaction edges** across 49 time steps, with labeled illicit transactions linked to scams, ransomware, and Ponzi schemes[2][3].
- **T-Finance:** A single-relation transaction network dataset containing **39,357 unique anonymized account nodes** and **21,222,543 transaction edges**[4][5]. Nodes are annotated by experts for fraud, money laundering, and online gambling (4.58% anomaly ratio)[6][7].
- **DGraph:** A massive real-world dynamic financial dataset modeling user interactions, loan-guarantor relations, and financial risks across millions of nodes[8].
- **FinFraudBench (CreditCard-Fraud &amp; BankTrans-Fraud):** Recent multi-entity, heterogeneous graph benchmarks constructed from real-world tabular financial records[11][12]. They include **6 node types** (*Transaction, Customer, Card, Merchant, Category, Location*) and **14 directed edge type**
- Elliptic2 (https://arxiv.org/abs/2404.19109). a large graph dataset containing 122K labeled subgraphs of Bitcoin clusters within a background graph consisting of 49M node clusters and 196M edge transactions. The dataset provides subgraphs known to be linked to illicit activity for learning the set of "shapes" that money laundering exhibits in cryptocurrency and accurately classifying new criminal activity