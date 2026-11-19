- This document is based on Aymen's reading and conclusions, as well as Claude's most notably [the recent papers' synthesis](./Synthesis_existing_reviews.md).

- Corpus overview (10 primary studies): already dealt with, included in report.

- General gaps and open questions in existing review papers identified by Aymen:
    - heterogenous graphs
    - subgraph level
    - rings
    - Imbalanced datasets
    - Delay (and difficulty) when labelling real world data 

- Gaps identified by Claude:
    - Gap: no subgraph / community-level detection partly closed for crypto and synthetic datasets. In banking data, collective detection is still a post-processing step on top of edge scores. Good, keep in final paper
    - Gap: patterns used only as edge labels. Ignore dont speak about it
    - Gap: static graphs dominate. temporal modeling is now the default. Good, but many different implementations with specfic techniques. Need to synthesize further.
    - Gap: heterogeneous graphs under-studied (M&R: ~30%).  Modest progress. Good, keep in final paper. Again specific techniques named (Transaction–entity graphs,  account–merchant graphs,...), which is probably too detailed or specific. "No study uses many entity types on a real banking graph with public results." <--- Can keep this sentence as is.
    - Gap: imbalance and inappropriate metrics → evaluation is becoming operational. Essentially Claude says progress is being made by ditching accuracy for metrics more adapted to imbalanced sets. Metrics used now: AP / PR-AUC, minority-class F1, P@k, Recall@k. Good to keep, they're famous metrics, mention by full name. Other custom metrics, jsut mention they're being used, but do not develop further. Draft sentence: "Previous review papers noted the issue of imbalanced datasets. Recent papers address it by using metrics other than accuracy, such Area Under Curve, F1-score, AP, and even custome metrics. The variety of metrics used make apple-to-apple comparisons more difficult accross diferent papers."
    - Deployment (latency, scale) → now measured. Seems minor and unimportant (space is becoming limited), ask Claude if its worth including in final paper
    - All these gaps and their resolutions should be included (if included) in the "Discussion" chapter

- Other notes from Claude synthesis:
 - 2.7 Mapping to the three typologies of the title. Interesting, but table seems to take a lot of space. Maybe just explain it in a few sentences. For example, "LAS-GNN and MEGA both have both been applied to illegal Rings and Structuring". in fact not really suer what this table says...What does "Strongest evidence" mean?

- New finding not reported by the reviews: GNNs do not consistently beat strong tabular models
    - In teh current version of my paper it's in the Results chapter. This point could be moved to the Disccusions instead. Results are supposed to be just for objective findings, not qualitative judgements. Still, the table of results showing when tabular models being comptetitive is worth showing. My issue with that is the subjective judgements such as "Trees win clearly" they should be reformulated.
    - Claude then argues that reports of GNN advantages use weak baselines, then goes into specifics in detail. Too detailed, I dont understand what it means. Ignore this part unless Claude can synthesize clearly.

- Results table: Already included in report. dealt with

- Evidence-quality grades: Seems out of my depth. Claude's arguments seem sound but I only read the paper's abstract's annd conclusions. I cant spend time to actually cross reference the venues. 2 options: report the papers 'results as is without judgement. Or exclude the C grades papers from my review. One of them (Gardezi)  is in fact already excluded.
- Errors and suspicions. This goes way too far in the details. I dont have the time or skill level to criticizse so many papers to such an extent. I should focus on the fact that my review paper does a synthesis. Still, I should ask Claude to synthesize even further and report the most critical issues.
- If I speak about the paper's quality, it should be in the methodology chapter probably

-  What remains open (input for Chapter 5)
    - "No real-bank benchmark with group labels. Elliptic2 is crypto, with heuristic labels; IBM AML and AMLSim are synthetic.". Very good, sbhould speak about this in the discussion chapter. Should I speak about the culture of traditional financial institutions (banks, etc) of secrecy (mostly justified to protect trade secrets, clients privacy and comptetive advantages)
    - "Structuring has no real labels." Need to investigate further, might be added to final paper.
    - "Group output is mostly post-hoc". Same as just above. Need to ask Claude for a further explanation of this topic
    - "Expressive pattern-aware GNNs are tested where patterns define the labels." I dont understant what this is talking about, better drop it.
    - "GNN vs gradient boosting is unresolved". Very interesting! SHould add to final paper. Need to ask for reference. Which papers talk about this?
    - "Unsupervised / label-free collective detection". Is this talking about Unsupervised learning? If yes very good point, add it to paper.
    - "Reproducibility ". Interesting, ask Claude to develop further.

- Content for the reviews section: Li et al. (2025). Interesting, but we're too far alon in the process to change the earlier chapters. Just ignore this paper unless their resuls are really significant

- Draft text for Chapter 4 (to adapt). In the process of being adapted. TO work on further.

- Follow-ups.Ok good to know. will be done later.