import argparse
import json
import logging
import pickle

import tqdm

from core.io.CacheManager import CacheManager
from core.io.ServerManager import ServerManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler

__author__ = "Pierre Monnin"


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (triplestore address, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--max-rows", dest="max_rows", help="Max number of rows returned by the SPARQL endpoint",
                        required=True, type=int, default=10000)
    parser.add_argument("--output", dest="output_dir", help="Base directory for output files", required=True)
    args = parser.parse_args()

    with open(args.conf_file_path, 'r') as configuration_file:
        configuration_parameters = json.load(configuration_file, encoding="utf-8")

    # Building ServerManager object and CacheManager
    server_manager = ServerManager(configuration_parameters, args.max_rows)
    cache_manager = CacheManager()

    # Logging parameters
    logger = logging.getLogger()
    tqdm_logging_handler = TqdmLoggingHandler()
    tqdm_logging_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s"))
    logger.addHandler(tqdm_logging_handler)
    logger.setLevel(logging.INFO)

    # Querying individuals
    logger.info("Querying individuals to reconcile")
    relations = set()
    for c in configuration_parameters["individuals-classes"]:
        for ind in server_manager.query_elements("?e rdf:type <{}>".format(c)):
            ind_i = cache_manager.get_element_index(ind)
            relations.add(ind_i)
    logger.info("Number of individuals to reconcile: " + str(len(relations)))

    # Querying similarity links between relations
    data_set = {}
    for l in tqdm.tqdm(configuration_parameters["similarity-links"]):
        data_set[l["url"]] = set()

        for c1 in configuration_parameters["individuals-classes"]:
            for c2 in configuration_parameters["individuals-classes"]:
                logger.info("Querying {} links for {} / {} ".format(l["url"], c1, c2))

                ret_val = server_manager.query_two_elements(
                    """
                    ?e1 rdf:type <{}> . 
                    ?e2 rdf:type <{}> .
                    ?e1 <{}> ?e2 .
                    """.format(c1, c2, l["url"]),
                    verbose=True
                )

                for ind1, ind2 in ret_val:
                    ind1_i = cache_manager.get_element_index(ind1)
                    ind2_i = cache_manager.get_element_index(ind2)

                    # When having (ind2_i, ind1_i) for a symmetric predicate, we avoid the symmetry bias
                    if not l["symmetry"] or (ind2_i, ind1_i) not in data_set[l["url"]]:
                        data_set[l["url"]].add((ind1_i, ind2_i))

    # Saving dataset
    logger.info("Saving simset")
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    cache_manager.save_to_csv(output_dir + "ind_cachemanager.csv")
    pickle.dump(data_set, open(output_dir + "ind_similaritylinks", "wb"))

    # Saving statistics
    with open(output_dir + "ind_statistics.csv", 'w') as file:
        file.write("# Individuals to reconcile,{}\n".format(len(relations)))

        for l in configuration_parameters["similarity-links"]:
            file.write("# links {},{}\n".format(l["url"], len(data_set[l["url"]])))


if __name__ == '__main__':
    main()
