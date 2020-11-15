# gcn-matching

Matching nodes in a knowledge graph using Graph Convolutional Networks and investigating the interplay between formal semantics and GCNs.

A detailed description of the motivation and the algorithms is available in [the related preprint](https://arxiv.org/pdf/2011.06023.pdf).

## Citing

When citing, please use the following reference:

Monnin, P., Raïssi, C., Napoli, A., & Coulet, A. (2020). Rediscovering alignment relations with Graph Convolutional Networks. arXiv preprint arXiv:2011.06023.

```
@misc{monnin2020rediscovering,
  title={{Rediscovering alignment relations with Graph Convolutional Networks}},
  author={Pierre Monnin and Chedy Raïssi and Amedeo Napoli and Adrien Coulet},
  year={2020},
  eprint={2011.06023},
  archivePrefix={arXiv}
}
```

## Scripts

### 1. Query similarity set

* In ``query_simset.py``
* Retrieve individuals to match (instances of classes in ``individuals-classes`` in the JSON configuration file)
* Retrieve similarity links between these individuals (to use in train/valid/test sets).
  * Similarity links are described in ``similarity-links`` in the JSON configuration file
  * When having the link ``(url1, url2)``, we do not add ``(url2, url1)`` for symmetric predicates to avoid the symmetry
  bias in training

### 2. Query graph

* In ``query_graph.py``
* Retrieve the adjacency of the RDF graph (except similarity links previously retrieved in 1.)
* Retrieve predicates and their inverses (or symmetry)
* **Must be used with the cache manager resulting from the previous step**

### 3. Similarity analysis

* In ``similarity_analysis.py``
* Output PDF files with histograms depicting the size of similarity clusters and number of them for each model (computed
based on the similarity links considered by each model)
* Similarity clusters for each model are computed over all similarity links indifferently considered by the model in an
 undirected (symmetry) and transitive fashion

### 4. N Fold Split

* In ``n_fold_split.py``
* Output a n-fold split of similarity links (after shuffling)

### 5. Transform graph

* In ``transform_graph.py``
* Output a DGL graph from the given RDF graph applying one of the following transformations:
  * G<sub>0</sub>: RDF graph + adding an abstract inverse for each predicate
  * G<sub>1</sub>: RDF graph after owl:sameAs edges contraction (only considering canonical nodes)
  * G<sub>2</sub>: RDF graph with consideration of inverse predicates / symmetry (to avoid adding abstract inverses when
  not needed)
  * G<sub>3</sub>: RDF graph with links added based on the hierarchy of predicates: if (a, rel<sub>1</sub>, b) and
  (rel<sub>1</sub>, subPropertyOf, rel<sub>2</sub>), we add (a, rel<sub>2</sub>, b)
  * G<sub>4</sub>: RDF graph with ``rdf:type`` links added based on the hierarchy of classes: if (a, type, b) and (b,
  subClassOf, c), we add (a, type, c)
  * G<sub>5</sub>:  all transformations of G<sub>1</sub> to G<sub>4</sub>
* The graph is limited to the considered neighborhood of individuals to match based on the number of layers

### 6. Learning

* In ``learning.py``
* Output a python dict where each key is the index of the test fold and contains:
  * ``logits_history``: python list associating an epoch with its logits
  * ``train_loss_history``: python list associating an epoch with its train loss
  * ``val_loss_history``: python list associating an epoch with its validation loss
  * ``test_loss_history``: python list associating an epoch with its test loss
  * ``temperature_history``: python list associating an epoch with its temperature
  * ``model``: python list associating an epoch with the parameters of the GCN model

### 7. Clustering analysis

* In ``clustering_analysis.py``
* Output for each fold:
  * A distance analysis based on all links, links whose nodes are in the training set, in the validation set, and in the test set
  * A UMAP projection computed on all nodes and displayed for all nodes, nodes in the training set, in the validation set and in the test set.
  Only ``--umap-colors`` similarity clusters are colored (starting at the biggest ones). Only similarity clusters containing more than ``--umap-size`` nodes are displayed (0 to disable)
  * A plot of the training, validation, and test losses
  * A plot of the temperature

## Dependencies

* Python3.7
* tqdm
* requests
* pytorch
* dgl
* matplotlib
* scikit-learn
* umap-learn
* pynndescent

## Experiments

### Models

(called gold clusterings in the preprint)

| Similarity links | owl:sameAs | skos:closeMatch | skos:relatedMatch | skos:related | skos:broadMatch |
|---------|---------|---------|---------|---------|---------|
| Properties | T / S | T / S | nT / S | nT / S | T / nS |
| M<sub>0</sub> | X | X | X | X | X |
| M<sub>1</sub> | X | X | X | X | |
| M<sub>2</sub> | X |  |  |  |  |
| M<sub>3</sub> |  | X |  |  |  |
| M<sub>4</sub> |  |  | X |  |  |
| M<sub>5</sub> |  |  |  | X |  |
| M<sub>6</sub> |  |  |  |  | X |

* **T**: transitivity
* **S**: symmetry
* **nT** : non-transitivity
* **nS** : non-symmetry
