import json
import pickle

import argparse
import logging

import dgl
import dgl.init
import dgl.data.utils
import torch
import tqdm

from core.io.CacheManager import CacheManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler


def compute_symmetry(adjacency):
    symmetric_adjacency = dict()

    for n1 in tqdm.tqdm(adjacency, desc="symmetric adjacency"):
        if n1 not in symmetric_adjacency:
            symmetric_adjacency[n1] = set()

        symmetric_adjacency[n1] |= adjacency[n1]

        for n2 in adjacency[n1]:
            if n2 not in symmetric_adjacency:
                symmetric_adjacency[n2] = set()

            symmetric_adjacency[n2].add(n1)

    return symmetric_adjacency


def compute_transitive_closure(adjacency):
    closed_adjacency = dict()

    for n1 in tqdm.tqdm(adjacency, desc="transitive closure"):
        closed_adjacency[n1] = set(adjacency[n1])
        to_expand = set(closed_adjacency[n1])

        while len(to_expand) != 0:
            new_to_expand = set()

            for n2 in to_expand:
                if n2 in adjacency:
                    new_to_expand |= (adjacency[n2] - closed_adjacency[n1])
                    closed_adjacency[n1] |= adjacency[n2]

            to_expand = new_to_expand

    return closed_adjacency


def build_canonical_graph(graph_adjacency, predicates_cache_manager, max_individuals_index, logger):
    same_as_i = predicates_cache_manager.get_element_index("http://www.w3.org/2002/07/owl#sameAs")

    if same_as_i not in graph_adjacency:
        return graph_adjacency

    # Complete (by symmetry and transitivity) owl:sameAs adjacency
    graph_adjacency[same_as_i] = compute_transitive_closure(compute_symmetry(graph_adjacency[same_as_i]))

    # Detect all nodes in graph
    nodes_to_canonicalize = set()
    for p_i in tqdm.tqdm(graph_adjacency, "nodes detection"):
        for n1 in graph_adjacency[p_i]:
            nodes_to_canonicalize.add(n1)
            nodes_to_canonicalize |= graph_adjacency[p_i][n1]

    # Nodes canonicalization
    rdf_to_canonical_node = dict()
    max_canonical_index = 0

    # - Nodes representing individuals to reconcile (failure if there are involved in the owl:sameAs adjacency)
    for i in range(max_individuals_index + 1):
        assert max_canonical_index == i

        if i in graph_adjacency[same_as_i]:
            logger.critical("Individual to reconcile involved in the owl:sameAs adjacency")
            exit(-1)

        rdf_to_canonical_node[i] = max_canonical_index
        max_canonical_index += 1
        nodes_to_canonicalize -= {i}

    # - Other nodes
    with tqdm.tqdm(len(nodes_to_canonicalize), desc="nodes canonicalization") as pbar:
        while len(nodes_to_canonicalize) != 0:
            n1 = nodes_to_canonicalize.pop()

            same_as_component = {n1}
            if n1 in graph_adjacency[same_as_i]:
                same_as_component |= graph_adjacency[same_as_i][n1]

            for n in same_as_component:
                rdf_to_canonical_node[n] = max_canonical_index
                nodes_to_canonicalize -= {n}

            max_canonical_index += 1
            pbar.update(len(same_as_component))

    # Adjacency canonicalization
    canonical_adjacency = dict()
    for p_i in tqdm.tqdm(graph_adjacency, desc="adjacency canonicalization"):
        if p_i != same_as_i:
            canonical_adjacency[p_i] = dict()

            for n1 in graph_adjacency[p_i]:
                nc1 = rdf_to_canonical_node[n1]

                if nc1 not in canonical_adjacency[p_i]:
                    canonical_adjacency[p_i][nc1] = set()

                for n2 in graph_adjacency[p_i][n1]:
                    nc2 = rdf_to_canonical_node[n2]
                    canonical_adjacency[p_i][nc1].add(nc2)

    return canonical_adjacency


def complete_inverse_adjacencies(graph_adjacency, predicates_inverses):
    new_adjacency = dict()

    for p_i in tqdm.tqdm(graph_adjacency):
        if p_i not in new_adjacency:
            new_adjacency[p_i] = dict()

        for n1 in graph_adjacency[p_i]:
            if n1 not in new_adjacency[p_i]:
                new_adjacency[p_i][n1] = set()

            for n2 in graph_adjacency[p_i][n1]:
                new_adjacency[p_i][n1].add(n2)

                if p_i in predicates_inverses:
                    for p_inv_i in predicates_inverses[p_i]:
                        if p_inv_i not in new_adjacency:
                            new_adjacency[p_inv_i] = dict()

                        if n2 not in new_adjacency[p_inv_i]:
                            new_adjacency[p_inv_i][n2] = set()

                        new_adjacency[p_inv_i][n2].add(n1)

    return new_adjacency


def complete_superproperties_adjacencies(graph_adjacency, predicates_cache_manager, nodes_cache_manager):
    subproperty_i = predicates_cache_manager.get_element_index("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")

    if subproperty_i in graph_adjacency:
        superproperties = compute_transitive_closure(graph_adjacency[subproperty_i])

        p_indices = set(graph_adjacency.keys())
        for p_i in tqdm.tqdm(p_indices, desc="predicate"):
            p_uri = predicates_cache_manager.get_element_from_index(p_i)
            p_node_i = nodes_cache_manager.get_element_index(p_uri)

            if p_node_i in superproperties:
                for sup_node_i in tqdm.tqdm(superproperties[p_node_i], desc="p superproperties"):
                    sup_uri = nodes_cache_manager.get_element_from_index(sup_node_i)
                    sup_i = predicates_cache_manager.get_element_index(sup_uri)

                    if sup_i not in graph_adjacency:
                        graph_adjacency[sup_i] = dict()

                    for n1 in graph_adjacency[p_i]:
                        if n1 not in graph_adjacency[sup_i]:
                            graph_adjacency[sup_i][n1] = set()

                        graph_adjacency[sup_i][n1] |= graph_adjacency[p_i][n1]

    return graph_adjacency


def complete_type_adjacency(graph_adjacency, predicates_cache_manager):
    subclass_i = predicates_cache_manager.get_element_index("http://www.w3.org/2000/01/rdf-schema#subClassOf")

    if subclass_i in graph_adjacency:
        superclasses = compute_transitive_closure(graph_adjacency[subclass_i])

        type_i = predicates_cache_manager.get_element_index("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        for n1 in tqdm.tqdm(graph_adjacency[type_i], desc="nodes"):
            n1_types = set(graph_adjacency[type_i][n1])

            for n2 in n1_types:
                if n2 in superclasses:
                    graph_adjacency[type_i][n1] |= superclasses[n2]

    return graph_adjacency


def build_inv_adjacency(graph_adjacency):
    inv_adjacency = dict()
    for p_i in tqdm.tqdm(graph_adjacency, desc="predicate"):
        inv_adjacency[p_i] = dict()

        for n1_i in tqdm.tqdm(graph_adjacency[p_i], desc="node"):
            for n2_i in graph_adjacency[p_i][n1_i]:
                if n2_i not in inv_adjacency[p_i]:
                    inv_adjacency[p_i][n2_i] = set()

                inv_adjacency[p_i][n2_i].add(n1_i)

    return inv_adjacency


def get_influencing_neighborhood(graph_adjacency, inv_adjacency, max_individuals_index, nb_hops):
    to_expand = set(range(max_individuals_index + 1))
    influencing_neighborhood = set(range(max_individuals_index + 1))

    for _ in tqdm.tqdm(range(0, nb_hops), desc="# expansion"):
        new_to_expand = set()

        for n_i in tqdm.tqdm(to_expand, desc="nodes expanding"):
            for p_i in graph_adjacency:
                if n_i in graph_adjacency[p_i]:
                    new_to_expand |= (graph_adjacency[p_i][n_i] - influencing_neighborhood)
                    influencing_neighborhood |= graph_adjacency[p_i][n_i]

            for p_i in inv_adjacency:
                if n_i in inv_adjacency[p_i]:
                    new_to_expand |= (inv_adjacency[p_i][n_i] - influencing_neighborhood)
                    influencing_neighborhood |= inv_adjacency[p_i][n_i]

        to_expand = new_to_expand

    return influencing_neighborhood


def clean_adjacency(adjacency, influencing_neighborhood):
    predicates = set(adjacency.keys())

    for p_i in tqdm.tqdm(predicates):
        nodes1 = set(adjacency[p_i].keys())

        for n1_i in nodes1:
            if n1_i not in influencing_neighborhood:
                del adjacency[p_i][n1_i]

            else:
                adjacency[p_i][n1_i] &= influencing_neighborhood

                if len(adjacency[p_i][n1_i]) == 0:
                    del adjacency[p_i][n1_i]

        if len(adjacency[p_i]) == 0:
            del adjacency[p_i]


def build_dgl_graph(graph_adjacency, inv_adjacency, max_individuals_index, logger, predicates_inverses=None):
    # Mappings old graph -> transformed graph
    nodes_mapping = CacheManager()
    for i in range(max_individuals_index + 1):
        assert i == nodes_mapping.get_element_index(i)
    predicates_mapping = CacheManager()

    # Preparing transformed graph edges
    edges_src = []
    edges_dest = []
    edges_rel_type = []
    edges_norm = []

    logger.info("-- Processing adjacency")
    for p_i in tqdm.tqdm(graph_adjacency, desc="predicates"):
        for n1_i in tqdm.tqdm(graph_adjacency[p_i], desc="nodes"):
            for n2_i in graph_adjacency[p_i][n1_i]:
                n1_tr_i = nodes_mapping.get_element_index(n1_i)
                n2_tr_i = nodes_mapping.get_element_index(n2_i)

                edges_src.append(n1_tr_i)
                edges_dest.append(n2_tr_i)
                edges_rel_type.append(predicates_mapping.get_element_index(p_i))
                edges_norm.append(1 / len(inv_adjacency[p_i][n2_i]))

    logger.info("-- Processing inverse adjacency")
    for p_i in tqdm.tqdm(inv_adjacency, desc="predicates"):
        if predicates_inverses is None or p_i not in predicates_inverses:
            for n1_i in tqdm.tqdm(inv_adjacency[p_i], desc="nodes"):
                for n2_i in inv_adjacency[p_i][n1_i]:
                    n1_tr_i = nodes_mapping.get_element_index(n1_i)
                    n2_tr_i = nodes_mapping.get_element_index(n2_i)

                    edges_src.append(n1_tr_i)
                    edges_dest.append(n2_tr_i)
                    edges_rel_type.append(predicates_mapping.get_element_index("inv_" + str(p_i)))
                    edges_norm.append(1 / len(graph_adjacency[p_i][n2_i]))

    logger.info("-- Adding self connection")
    for n_i in tqdm.tqdm(range(nodes_mapping.get_size())):
        edges_src.append(n_i)
        edges_dest.append(n_i)
        edges_rel_type.append(predicates_mapping.get_element_index("self-connection"))
        edges_norm.append(1)

    # Building DGL graph object
    logger.info("-- Creating the DGL graph object")
    g = dgl.DGLGraph(multigraph=True)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    g.add_nodes(nodes_mapping.get_size())
    g.add_edges(edges_src, edges_dest)
    edges_rel_type = torch.LongTensor(edges_rel_type)
    edges_norm = torch.Tensor(edges_norm).unsqueeze(1)
    g.edata.update({
        "rel_type": edges_rel_type,
        "norm": edges_norm
    })

    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (layers, size of embeddings, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--ind-cache-manager", help="CacheManager file for individuals to reconcile",
                        required=True, dest="ind_cache_manager_file_path")
    parser.add_argument("--predicates-cache-manager", help="CacheManager file for predicates in the RDF graph",
                        required=True, dest="predicates_cache_manager_file_path")
    parser.add_argument("--predicates-inverses", help="File containing predicates inverses", required=True,
                        dest="predicates_inverses_file_path")
    parser.add_argument("--rdf-adjacency", help="File containing the RDF graph adjacency", required=True,
                        dest="rdf_adjacency_file_path")
    parser.add_argument("--nodes-cache-manager", help="CacheManager file for nodes in the RDF graph",
                        required=True, dest="nodes_cache_manager_file_path")
    parser.add_argument("--graph-transform", help="Graph transformation to perform (see README)", type=int,
                        choices=[0, 1, 2, 3, 4, 5], default=0, dest="graph_transform")
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    args = parser.parse_args()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Loading configuration
    logger.info("Loading configuration")
    with open(args.conf_file_path, 'r') as configuration_file:
        configuration_parameters = json.load(configuration_file, encoding="utf-8")

    # Loading CacheManager for individuals to reconcile
    logger.info("Loading CacheManager for individuals to reconcile")
    ind_cache_manager = CacheManager()
    ind_cache_manager.load_from_csv(args.ind_cache_manager_file_path)

    # Loading CacheManager for predicates in the RDF graph
    logger.info("Loading CacheManager for predicates in the RDF graph")
    predicates_cache_manager = CacheManager()
    predicates_cache_manager.load_from_csv(args.predicates_cache_manager_file_path)

    # Loading predicates inverses
    logger.info("Loading predicates inverses")
    predicates_inverses = pickle.load(open(args.predicates_inverses_file_path, "rb"))

    # Loading RDF graph adjacency
    logger.info("Loading RDF graph adjacency")
    graph_adjacency = pickle.load(open(args.rdf_adjacency_file_path, "rb"))

    # Loading CacheManager for nodes in the RDF graph
    logger.info("Loading CacheManager for nodes in the RDF graph")
    nodes_cache_manager = CacheManager()
    nodes_cache_manager.load_from_csv(args.nodes_cache_manager_file_path)

    if args.graph_transform == 1 or args.graph_transform == 5:
        logger.info("Compute canonical graph (owl:sameAs edges contraction)")
        graph_adjacency = build_canonical_graph(
            graph_adjacency,
            predicates_cache_manager,
            ind_cache_manager.get_size() - 1,
            logger
        )

    if args.graph_transform == 2 or args.graph_transform == 5:
        logger.info("Completing adjacencies based on inverse and symmetric predicates")
        graph_adjacency = complete_inverse_adjacencies(
            graph_adjacency,
            predicates_inverses
        )

    if args.graph_transform == 3 or args.graph_transform == 5:
        # Computing transitive closure of subProperty edges and adding missing edges accordingly
        logger.info("Completing adjacency based on subsumption of properties")
        graph_adjacency = complete_superproperties_adjacencies(
            graph_adjacency,
            predicates_cache_manager,
            nodes_cache_manager
        )

    if args.graph_transform == 4 or args.graph_transform == 5:
        # Computing transitive closure of instantiation edges
        logger.info("Completing rdf:type edges based on subsumption of classes")
        graph_adjacency = complete_type_adjacency(
            graph_adjacency,
            predicates_cache_manager
        )

    # Building inverse adjacency
    logger.info("Building graph inverse adjacency")
    inv_adjacency = build_inv_adjacency(graph_adjacency)

    # Computing influencing neighborhood
    logger.info("Computing influencing neighborhood")
    influencing_neighborhood = get_influencing_neighborhood(
        graph_adjacency,
        inv_adjacency,
        ind_cache_manager.get_size(),
        configuration_parameters["hidden-layers"] + 2
    )

    # Removing useless nodes and predicates from adjacencies
    logger.info("Removing useless nodes and predicates from adjacencies")
    clean_adjacency(graph_adjacency, influencing_neighborhood)
    clean_adjacency(inv_adjacency, influencing_neighborhood)

    # Building DGL graph
    logger.info("Building DGL graph")
    if args.graph_transform == 2 or args.graph_transform == 5:
        g = build_dgl_graph(
            graph_adjacency,
            inv_adjacency,
            ind_cache_manager.get_size() - 1,
            logger,
            predicates_inverses
        )
    else:
        g = build_dgl_graph(
            graph_adjacency,
            inv_adjacency,
            ind_cache_manager.get_size() - 1,
            logger
        )

    # Preparing output dir
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    # Saving DGL graph
    logger.info("Saving transformed graph in DGL format")
    dgl.data.utils.save_graphs(output_dir + "graph_{}".format(args.graph_transform), g)

    # Saving transformed graph statistics
    logger.info("Saving transformed graph statistics")
    with open(output_dir + "graph_{}_statistics.md".format(args.graph_transform), "w") as file:
        file.write("* Number of nodes: {}\n".format(g.number_of_nodes()))
        file.write("* Number of edges: {}\n".format(g.number_of_edges()))
        file.write("* Number of predicates: {}\n".format(g.edata["rel_type"].max().item() + 1))


if __name__ == '__main__':
    main()
