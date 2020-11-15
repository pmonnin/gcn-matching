import argparse
import json
import logging
import pickle

import numpy
import sklearn.model_selection

from core.learning.utils import from_similarity_clusters_to_labels
from core.io.CacheManager import CacheManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (similarity links, models, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--ind-cache-manager", help="CacheManager of individuals to reconcile", required=True,
                        dest="ind_cache_manager")
    parser.add_argument("--model", help="Model whose similarity clusters are considered in a stratified k-fold",
                        type=int, required=True)
    parser.add_argument("--similarity-clusters", help="Similarity clusters computed by similarity_analysis.py",
                        required=True, dest="similarity_clusters_file_path")
    parser.add_argument("--nfolds", help="Number of folds", required=True, dest="nfolds", type=int)
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

    min_model = min([m for l in configuration_parameters["similarity-links"] for m in l["models"]])
    max_model = max([m for l in configuration_parameters["similarity-links"] for m in l["models"]])
    if args.model < min_model or args.model > max_model:
        logger.critical("Invalid number of model: {} (expected between {} and {})".format(args.model, min_model,
                                                                                          max_model))
        exit(-1)

    # Loading cache manager
    logger.info("Loading CacheManager of individuals to reconcile")
    cache_manager = CacheManager()
    cache_manager.load_from_csv(args.ind_cache_manager)
    logger.info("{} individuals to reconcile".format(cache_manager.get_size()))

    # Loading similarity clusters
    logger.info("Loading similarity clusters")
    similarity_clusters = pickle.load(open(args.similarity_clusters_file_path, "rb"))
    similarity_labels = from_similarity_clusters_to_labels(similarity_clusters[args.model], cache_manager.get_size())

    # Shuffling and splitting individuals to reconcile
    logger.info("Generating folds")
    kfold = sklearn.model_selection.StratifiedKFold(args.nfolds, shuffle=True)
    folds = []
    for _, test_inds in kfold.split(numpy.zeros((cache_manager.get_size(),)), similarity_labels):
        folds.append(test_inds)
        logger.info("Fold {} contains {} individuals".format(len(folds) - 1, len(test_inds)))

    # Saving folds
    logger.info("Saving folds")

    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    pickle.dump(folds, open(output_dir + "{}_folds".format(args.nfolds), "wb"))


if __name__ == '__main__':
    main()
