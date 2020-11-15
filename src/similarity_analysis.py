import argparse
import json
import logging
import pickle
import queue

import matplotlib.pyplot
import tqdm

from core.io.CacheManager import CacheManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler


def compute_similarity_cluster(n, adjacency):
    q = queue.Queue()
    q.put(n)
    similarity_cluster = {n}

    while not q.empty():
        current_node = q.get()

        if current_node in adjacency:
            for neighbor in adjacency[current_node]:
                if neighbor not in similarity_cluster:
                    similarity_cluster.add(neighbor)
                    q.put(neighbor)

    return similarity_cluster


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (models, similarity links, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--ind-cache-manager", help="CacheManager of individuals to reconcile", required=True,
                        dest="ind_cache_manager")
    parser.add_argument("--similarity-links", help="File containing the similarity links to analyze", required=True,
                        dest="similarity_links_file_path")
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    parser.add_argument("--figsize-x", dest="figsize_x", help="X size of matplotlib figures", default=20)
    parser.add_argument("--figsize-y", dest="figsize_y", help="Y size of matplotlib figures", default=10)
    args = parser.parse_args()

    with open(args.conf_file_path, 'r') as configuration_file:
        configuration_parameters = json.load(configuration_file, encoding="utf-8")

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Loading cache manager
    logger.info("Loading CacheManager of individuals to reconcile")
    cache_manager = CacheManager()
    cache_manager.load_from_csv(args.ind_cache_manager)

    # Loading similarity links
    logger.info("Loading similarity links")
    similarity_links = pickle.load(open(args.similarity_links_file_path, "rb"))

    # Detecting number of models
    number_of_models = max([i for l in configuration_parameters["similarity-links"] for i in l["models"]])
    logger.info("Number of models detected: {}".format(number_of_models))

    # Computing similarity clusters statistics
    similarity_clusters = dict()
    similarity_clusters_statistics = dict()
    for model in tqdm.tqdm(range(0, number_of_models + 1)):
        similarity_clusters[model] = list()
        similarity_clusters_statistics[model] = dict()

        # Computing similarity adjacency for the current model
        adjacency = dict()
        for l in configuration_parameters["similarity-links"]:
            if model in l["models"]:
                for rel1_i, rel2_i in similarity_links[l["url"]]:
                    # We consider that similarity links are symmetric for the similarity cluster computation
                    if rel1_i not in adjacency:
                        adjacency[rel1_i] = set()
                    adjacency[rel1_i].add(rel2_i)

                    if rel2_i not in adjacency:
                        adjacency[rel2_i] = set()
                    adjacency[rel2_i].add(rel1_i)

        # Computing similarity clusters for the current model
        nodes = set(range(cache_manager.get_size()))
        with tqdm.tqdm(total=len(nodes)) as pbar:
            while len(nodes) != 0:
                n = nodes.pop()
                cluster = compute_similarity_cluster(n, adjacency)

                pbar.update(len(cluster))
                nodes = nodes - cluster
                similarity_clusters[model].append(cluster)

                if len(cluster) not in similarity_clusters_statistics[model]:
                    similarity_clusters_statistics[model][len(cluster)] = 0
                similarity_clusters_statistics[model][len(cluster)] += 1

    # Saving results
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    pickle.dump(similarity_clusters, open(output_dir + "similarity_clusters", "wb"))
    pickle.dump(similarity_clusters_statistics, open(output_dir + "similarity_clusters_statistics", "wb"))

    for model in similarity_clusters_statistics:
        indices = range(0, len(similarity_clusters_statistics[model]))
        x_labels = list(similarity_clusters_statistics[model].keys())
        x_labels.sort()
        y_values = [similarity_clusters_statistics[model][x] for x in x_labels]

        fig = matplotlib.pyplot.figure(figsize=(args.figsize_x, args.figsize_y))
        matplotlib.pyplot.bar(
            indices,
            y_values,
            log=True
        )
        fig.suptitle("Model {}".format(model))
        matplotlib.pyplot.xlabel("Size of similarity clusters")
        matplotlib.pyplot.ylabel("Number of similarity clusters")
        matplotlib.pyplot.xticks(indices, x_labels, rotation="vertical")
        matplotlib.pyplot.savefig(output_dir + "M" + str(model) + "_statistics.pdf", bbox_inches="tight")


if __name__ == '__main__':
    main()
