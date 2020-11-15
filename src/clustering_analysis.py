import argparse
import json
import logging
import pickle
import statistics

import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import numpy
import numpy.linalg
import sklearn
import sklearn.cluster
import sklearn.metrics
import tqdm
import umap

from core.io.CacheManager import CacheManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler
from core.learning.utils import from_similarity_clusters_to_colors, size_constrained_clusters_to_labels

url_mapping = {
    "http://www.w3.org/2002/07/owl#sameAs": "sameAs",
    "http://www.w3.org/2004/02/skos/core#closeMatch": "closeMatch",
    "http://www.w3.org/2004/02/skos/core#relatedMatch": "relatedMatch",
    "http://www.w3.org/2004/02/skos/core#related": "related",
    "http://www.w3.org/2004/02/skos/core#broadMatch": "broadMatch"
}

learning_set_to_color = {
    "training": "blue",
    "validation": "orange",
    "test": "red",
    "all": "green"
}


def save_umap_plot(umap_result, nodes_to_color, node_colors, test_fold, umap_colors, umap_size, figsize_x, figsize_y,
                   name, output_dir):
    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.scatter(
        umap_result[nodes_to_color, 0],
        umap_result[nodes_to_color, 1],
        cmap="gist_rainbow",
        c=node_colors[nodes_to_color],
        alpha=0.5,
        s=10
    )
    fig.suptitle("UMAP projection for test fold {} - {} nodes - {} clusters - max size {}".format(
        test_fold, name, umap_colors, umap_size if umap_size > 0 else "disabled"))
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_" + name + "_umap.pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def umap_projections(test_fold, logits, folds, similarity_clusters, individuals_number, umap_colors, umap_size,
                     figsize_x, figsize_y, output_dir):
    node_colors = from_similarity_clusters_to_colors(similarity_clusters, individuals_number, umap_colors, umap_size)
    nodes_to_color = [i for i, c in enumerate(node_colors) if c != 0]

    test_set = sorted(folds[test_fold])
    test_set_to_color = sorted(list(set(test_set) & set(nodes_to_color)))
    val_set = sorted(folds[(test_fold + 1) % len(folds)])
    val_set_to_color = sorted(list(set(val_set) & set(nodes_to_color)))
    train_set = sorted(list(set().union(*[set(f) for f in folds]) - set(test_set) - set(val_set)))
    train_set_to_color = sorted(list(set(train_set) & set(nodes_to_color)))

    umap_result = umap.UMAP(
        n_neighbors=50,
        min_dist=0.001,
        metric='euclidean',
        n_components=2
    ).fit_transform(logits)

    # UMAP global projection
    save_umap_plot(
        umap_result,
        nodes_to_color,
        node_colors,
        test_fold,
        umap_colors,
        umap_size,
        figsize_x,
        figsize_y,
        "all",
        output_dir
    )

    # UMAP training set projection
    save_umap_plot(
        umap_result,
        train_set_to_color,
        node_colors,
        test_fold,
        umap_colors,
        umap_size,
        figsize_x,
        figsize_y,
        "training",
        output_dir
    )

    # UMAP validation set projection
    save_umap_plot(
        umap_result,
        val_set_to_color,
        node_colors,
        test_fold,
        umap_colors,
        umap_size,
        figsize_x,
        figsize_y,
        "validation",
        output_dir
    )

    # UMAP test set projection
    save_umap_plot(
        umap_result,
        test_set_to_color,
        node_colors,
        test_fold,
        umap_colors,
        umap_size,
        figsize_x,
        figsize_y,
        "test",
        output_dir
    )


def save_losses_plot(train_loss_history, val_loss_history, test_loss_history, test_fold, figsize_x, figsize_y,
                     output_dir):
    indices = range(0, len(train_loss_history))
    x_labels = list(indices)
    x_labels.sort()

    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.plot(
        indices,
        train_loss_history,
        color="blue",
        label="Train loss"
    )
    matplotlib.pyplot.plot(
        indices,
        val_loss_history,
        color="orange",
        linestyle='dashed',
        label="Validation loss"
    )
    matplotlib.pyplot.plot(
        indices,
        test_loss_history,
        color="red",
        linestyle='dotted',
        label="Test loss"
    )
    fig.suptitle("Evolution of losses for test fold {}".format(test_fold))
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss value")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.xticks([i for i in indices if i % 10 == 0],
                             [l for i, l in enumerate(x_labels) if i % 10 == 0], rotation="vertical")
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_losses.pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def save_temperature_plot(temperature_history, test_fold, figsize_x, figsize_y, output_dir):
    indices = range(0, len(temperature_history))
    x_labels = list(indices)
    x_labels.sort()

    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.plot(
        indices,
        temperature_history,
        color="green",
        label="Temperature"
    )
    fig.suptitle("Evolution of the temperature for test fold {}".format(test_fold))
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Temperature value")
    matplotlib.pyplot.legend(loc="best")
    matplotlib.pyplot.xticks([i for i in indices if i % 10 == 0],
                             [l for i, l in enumerate(x_labels) if i % 10 == 0], rotation="vertical")
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_temperature.pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def save_distance_hist(distances, learning_set_name, test_fold, pred, figsize_x, figsize_y, output_dir):
    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.hist(distances, color=learning_set_to_color[learning_set_name], log=True)
    fig.suptitle("Distribution of distances in {} links for test fold {} - {} nodes".format(
        pred, test_fold, learning_set_name))
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_distances_" + pred + "_" + learning_set_name
                              + ".pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def save_distance_box_plot(distances, labels, learning_set_name, test_fold, figsize_x, figsize_y, output_dir):
    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.boxplot(distances, labels=labels, whis=10000000)
    fig.suptitle("Distribution of distances per similarity predicate for test fold {} - {} nodes".format(
        test_fold, learning_set_name))
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_distances_" + learning_set_name
                              + ".pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def distance_analysis(logits, test_fold, folds, similarity_links, figsize_x, figsize_y, output_dir):
    test_set = set(folds[test_fold])
    val_set = set(folds[(test_fold + 1) % len(folds)])
    train_set = set().union(*[set(f) for f in folds]) - test_set - val_set

    distances = dict()

    for l in tqdm.tqdm(similarity_links):
        distances[l] = {
            "training": list(),
            "validation": list(),
            "test": list(),
            "all": list()
        }

        for n1, n2 in tqdm.tqdm(similarity_links[l]):
            d = numpy.linalg.norm(logits[n1, :] - logits[n2, :])

            if n1 in test_set and n2 in test_set:
                distances[l]["test"].append(d)
            elif n1 in val_set and n2 in val_set:
                distances[l]["validation"].append(d)
            elif n1 in train_set and n2 in train_set:
                distances[l]["training"].append(d)

            distances[l]["all"].append(d)

        for d in distances[l]:
            save_distance_hist(distances[l][d], d, test_fold, url_mapping[l], figsize_x, figsize_y, output_dir)

    for learning_set_name in ["training", "validation", "test", "all"]:
        dist_list = []
        labels = []

        for l in similarity_links:
            dist_list.append(distances[l][learning_set_name])
            labels.append(url_mapping[l])

        save_distance_box_plot(dist_list, labels, learning_set_name, test_fold, figsize_x, figsize_y, output_dir)

    with open(output_dir + "F" + str(test_fold) + "_distances_analysis.md", "w") as markdown_file:
        markdown_file.write("| Link | Train set | Validation set | Test set | Global |\n")
        markdown_file.write("|------|-----------|----------------|----------|--------|\n")
        for l in distances:
            test_distances = numpy.array(distances[l]["test"])
            val_distances = numpy.array(distances[l]["validation"])
            train_distances = numpy.array(distances[l]["training"])
            global_distances = numpy.array(distances[l]["all"])

            output_string = "| {} |".format(url_mapping[l])
            for d in [train_distances, val_distances, test_distances, global_distances]:
                output_string += " Mean: {}; Std: {}; Min: {}; Max: {} |".format(d.mean(), d.std(), d.min(), d.max())
            markdown_file.write(output_string + "\n")


def relabel(labels):
    retval = numpy.zeros(labels.shape, dtype=numpy.int64)
    mapping = dict()

    new_label = 0
    for i, l in enumerate(labels):
        if l not in mapping:
            mapping[l] = new_label
            new_label += 1

        retval[i] = mapping[l]

    return retval


def unsupervised_clustering_accuracy(labels_true, labels_pred):
    """
    Calculate the unsupervised clustering accuracy.
    # Arguments
        labels_true: true labels, numpy.array with shape `(n_samples,)`
        labels_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    Adapted from https://github.com/XifengGuo/DEC-keras/blob/master/metrics.py
    """
    y_true = relabel(labels_true)
    y_pred = relabel(labels_pred)
    assert y_true.size == y_pred.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = numpy.zeros((D, D), dtype=numpy.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_performance(logits, test_fold, folds, similarity_clusters, similarity_links, min_cluster_size,
                           individuals_number, output_dir):
    test_set = set(folds[test_fold])
    val_set = set(folds[(test_fold + 1) % len(folds)])
    train_set = set().union(*[set(f) for f in folds]) - test_set - val_set

    labels = size_constrained_clusters_to_labels(similarity_clusters, individuals_number, min_cluster_size)
    all_labelled_nodes = set(numpy.argwhere(labels != -1)[:, 0])
    labelled_nodes = {
        "Training nodes": sorted(train_set & all_labelled_nodes),
        "Validation nodes": sorted(val_set & all_labelled_nodes),
        "Test nodes": sorted(test_set & all_labelled_nodes),
        "All nodes": sorted(all_labelled_nodes)
    }

    mapping_labelled_nodes = dict()
    for k in tqdm.tqdm(labelled_nodes):
        mapping_labelled_nodes[k] = dict()
        for i, n in tqdm.tqdm(enumerate(labelled_nodes[k])):
            mapping_labelled_nodes[k][n] = i

    labelled_links = dict()
    for l in tqdm.tqdm(similarity_links, "link prep."):
        labelled_links[l] = {
            "Training nodes": [],
            "Validation nodes": [],
            "Test nodes": [],
            "All nodes": []
        }

        for n1, n2 in tqdm.tqdm(similarity_links[l]):
            for k in mapping_labelled_nodes:
                if n1 in mapping_labelled_nodes[k] and n2 in mapping_labelled_nodes[k]:
                    n1_i = mapping_labelled_nodes[k][n1]
                    n2_i = mapping_labelled_nodes[k][n2]
                    labelled_links[l][k].append((n1_i, n2_i))

    optics_min_samples = {
        "Training nodes": int(min_cluster_size * (len(folds) - 1) / len(folds)),
        "Validation nodes": int(min_cluster_size / len(folds)),
        "Test nodes": int(min_cluster_size / len(folds)),
        "All nodes": int(min_cluster_size)
    }

    clustering_metrics = {
        "Ward": dict(),
        "Single": dict(),
        "OPTICS": dict()
    }

    links_metrics = {
        "Ward": {l: dict() for l in similarity_links},
        "Single": {l: dict() for l in similarity_links},
        "OPTICS": {l: dict() for l in similarity_links}
    }

    for k in tqdm.tqdm(labelled_nodes, desc="label set"):
        clustering_algorithm = {
            "Ward": sklearn.cluster.AgglomerativeClustering(n_clusters=labels.max()),
            "Single": sklearn.cluster.AgglomerativeClustering(n_clusters=labels.max(), linkage="single"),
            "OPTICS": sklearn.cluster.OPTICS(min_samples=optics_min_samples[k])
        }

        for ca in tqdm.tqdm(clustering_algorithm, desc="clustering alg."):
            predicted_labels = clustering_algorithm[ca].fit_predict(logits[labelled_nodes[k]])

            clustering_metrics[ca][k] = {
                "ACC": unsupervised_clustering_accuracy(labels[labelled_nodes[k]], predicted_labels),
                "ARI": sklearn.metrics.adjusted_rand_score(labels[labelled_nodes[k]], predicted_labels),
                "NMI": sklearn.metrics.adjusted_mutual_info_score(labels[labelled_nodes[k]], predicted_labels)
            }

            for l in tqdm.tqdm(similarity_links, desc="sim links"):
                links_metrics[ca][l][k] = 0
                for n1, n2 in tqdm.tqdm(labelled_links[l][k]):
                    if predicted_labels[n1] == predicted_labels[n2]:
                        links_metrics[ca][l][k] += 1

    with open(output_dir + "F{}_clustering_performance_{}_size.md".format(test_fold, min_cluster_size), "w") as file:
        for ca in clustering_metrics:
            file.write("# {}\n".format(ca))

            file.write("## Clustering Performance Metrics\n")
            file.write("Only nodes involved in clusters whose size is >= {} are considered.\n".format(min_cluster_size))
            file.write("| Nodes | ACC | ARI | NMI |\n")
            file.write("|-------|-----|-----|-----|\n")
            for k in ["Training nodes", "Validation nodes", "Test nodes", "All nodes"]:
                file.write("| {} | {} | {} | {} |\n".format(
                    k,
                    clustering_metrics[ca][k]["ACC"],
                    clustering_metrics[ca][k]["ARI"],
                    clustering_metrics[ca][k]["NMI"]
                ))

            file.write("## Link Performance Metrics\n")
            file.write("Only links between nodes involved in clusters whose size is >= {} are considered.\n".format(
                min_cluster_size))
            file.write("| Link | Training nodes  | Validation Nodes | Test nodes | All nodes |\n")
            file.write("|------|-----------------|------------------|------------|-----------|\n")
            for l in similarity_links:
                row = "| {} |".format(url_mapping[l])
                for k in ["Training nodes", "Validation nodes", "Test nodes", "All nodes"]:
                    if len(labelled_links[l][k]) != 0:
                        row += " {} / {} ({:.2f} %) |".format(
                            links_metrics[ca][l][k],
                            len(labelled_links[l][k]),
                            links_metrics[ca][l][k] / len(labelled_links[l][k]) * 100.0
                        )
                    else:
                        row += " n/a |"
                file.write(row + "\n")

            file.write("\n\n")

    return clustering_metrics


def embedding_values_histogram(logits, test_fold, figsize_x, figsize_y, output_dir):
    fig = matplotlib.pyplot.figure(figsize=(figsize_x, figsize_y))
    matplotlib.pyplot.hist(logits.flatten(), log=True)
    fig.suptitle("Distribution of values in embeddings - test fold {}".format(test_fold))
    matplotlib.pyplot.savefig(output_dir + "F" + str(test_fold) + "_embeddings_values.pdf", bbox_inches="tight")
    matplotlib.pyplot.close()


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (layers, size of embeddings, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--ind-cache-manager", help="CacheManager of individuals to reconcile", required=True,
                        dest="ind_cache_manager_file_path")
    parser.add_argument("--learning-results", help="File containing the learning results", required=True,
                        dest="learning_results_file_path")
    parser.add_argument("--model", help="Model to learn based on models defined in the configuration file", type=int,
                        required=True)
    parser.add_argument("--folds", help="File containing the similarity folds", required=True,
                        dest="similarity_folds_file_path")
    parser.add_argument("--umap-colors", dest="umap_colors", help="Number of clusters to color in UMAP", type=int,
                        default=30)
    parser.add_argument("--umap-size", dest="umap_size", help="Max size of clusters to color in UMAP", type=list,
                        default=0)
    parser.add_argument("--min-cluster-size", dest="min_cluster_size", help="Minimum size of clusters for evaluation",
                        type=int, nargs='+', required=True)
    parser.add_argument("--similarity-clusters", help="Similarity clusters computed by similarity_analysis.py",
                        required=True, dest="similarity_clusters_file_path")
    parser.add_argument("--similarity-links", help="File containing the similarity links to analyze", required=True,
                        dest="similarity_links_file_path")
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    parser.add_argument("--figsize-x", dest="figsize_x", help="X size of matplotlib figures", default=20)
    parser.add_argument("--figsize-y", dest="figsize_y", help="Y size of matplotlib figures", default=10)
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

    min_model = min([m for l in configuration_parameters["similarity-links"] for m in l["models"]])
    max_model = max([m for l in configuration_parameters["similarity-links"] for m in l["models"]])
    if args.model < min_model or args.model > max_model:
        logger.critical("Invalid number of model: {} (expected between {} and {})".format(args.model, min_model,
                                                                                          max_model))
        exit(-1)

    # Loading CacheManager for individuals to reconcile
    logger.info("Loading CacheManager for individuals to reconcile")
    ind_cache_manager = CacheManager()
    ind_cache_manager.load_from_csv(args.ind_cache_manager_file_path)

    # Loading learning results
    logger.info("Loading learning results")
    learning_results = pickle.load(open(args.learning_results_file_path, "rb"))

    # Loading similarity links
    logger.info("Loading similarity links")
    similarity_links = pickle.load(open(args.similarity_links_file_path, "rb"))

    # Loading similarity clusters
    logger.info("Loading similarity clusters")
    similarity_clusters = pickle.load(open(args.similarity_clusters_file_path, "rb"))

    # Loading folds
    logger.info("Loading folds")
    folds = pickle.load(open(args.similarity_folds_file_path, "rb"))

    # Preparing output directory
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    global_clustering_metrics = {size: [] for size in args.min_cluster_size if size > 0}

    # Computing analysis for each fold
    logger.info("Computing analysis for each fold")
    for test_fold in tqdm.tqdm(learning_results):
        # Number of epochs
        with open(output_dir + "F{}_epochs.md".format(test_fold), "w") as file:
            file.write("Number of epochs: {}\n".format(len(learning_results[test_fold]["logits_history"])))

        # Performance of clustering
        for size in args.min_cluster_size:
            if size > 0:
                logger.info("Performance of clustering on clusters with size >= {}".format(size))
                global_clustering_metrics[size].append(clustering_performance(
                    learning_results[test_fold]["logits_history"][-1],
                    test_fold,
                    folds,
                    similarity_clusters[args.model],
                    similarity_links,
                    size,
                    ind_cache_manager.get_size(),
                    output_dir
                ))

        # Distance analysis
        logger.info("Distance analysis for current fold")
        distance_analysis(
            learning_results[test_fold]["logits_history"][-1],
            test_fold,
            folds,
            similarity_links,
            args.figsize_x,
            args.figsize_y,
            output_dir
        )

        # Embedding values hist
        logger.info("Embedding values histogram")
        embedding_values_histogram(
            learning_results[test_fold]["logits_history"][-1],
            test_fold,
            args.figsize_x,
            args.figsize_y,
            output_dir
        )

        # UMAP projections
        logger.info("UMAP projections")
        umap_projections(
            test_fold,
            learning_results[test_fold]["logits_history"][-1],
            folds,
            similarity_clusters[args.model],
            ind_cache_manager.get_size(),
            args.umap_colors,
            args.umap_size,
            args.figsize_x,
            args.figsize_y,
            output_dir
        )

        # Losses figure
        logger.info("Evolution of losses")
        save_losses_plot(
            learning_results[test_fold]["train_loss_history"],
            learning_results[test_fold]["val_loss_history"],
            learning_results[test_fold]["test_loss_history"],
            test_fold,
            args.figsize_x,
            args.figsize_y,
            output_dir
        )

        # Temperature figure
        logger.info("Evolution of the temperature")
        save_temperature_plot(
            learning_results[test_fold]["temperature_history"],
            test_fold,
            args.figsize_x,
            args.figsize_y,
            output_dir
        )

    # Output global clustering metrics
    with open(output_dir + "global_clustering_metrics.md", "w") as file:
        for size in args.min_cluster_size:
            if size > 0:
                file.write("# Clusters whose size >= {}\n".format(size))
                for ca in ["Ward", "Single", "OPTICS"]:
                    file.write("## {}\n".format(ca))
                    file.write("| Nodes | ACC | ARI | NMI |\n")
                    file.write("|-------|-----|-----|-----|\n")
                    for k in ["Training nodes", "Validation nodes", "Test nodes", "All nodes"]:
                        file.write("| {} | {} \\pm {} | {} \\pm {} | {} \\pm {} |\n".format(
                            k,
                            statistics.mean([m[ca][k]["ACC"] for m in global_clustering_metrics[size]]),
                            statistics.stdev([m[ca][k]["ACC"] for m in global_clustering_metrics[size]]),
                            statistics.mean([m[ca][k]["ARI"] for m in global_clustering_metrics[size]]),
                            statistics.stdev([m[ca][k]["ARI"] for m in global_clustering_metrics[size]]),
                            statistics.mean([m[ca][k]["NMI"] for m in global_clustering_metrics[size]]),
                            statistics.stdev([m[ca][k]["NMI"] for m in global_clustering_metrics[size]]),
                        ))
                    file.write("\n\n")


if __name__ == '__main__':
    main()
