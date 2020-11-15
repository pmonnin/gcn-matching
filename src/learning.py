import argparse
import json
import logging
import pickle

import dgl.data.utils
import numpy
import torch.optim
import tqdm

from core.gcn.rgcnt import RGCNTModel
from core.io.CacheManager import CacheManager
from core.io.TqdmLoggingHandler import TqdmLoggingHandler
from core.learning.utils import from_similarity_clusters_to_mapping, from_learning_set_to_dict, \
    from_similarity_clusters_to_labels


STABILITY_EPS = 0.00001


def soft_nearest_neighbor_loss(logits, labels, temperature):
    b = labels.shape[0]

    same_label_mask = labels.expand(b, b).eq(labels.t().expand(b, b)).type(torch.FloatTensor)

    sqr_norm = logits.pow(2).sum(dim=1).unsqueeze(0)
    pairwise_euclid_distance = sqr_norm.expand(b, b) + sqr_norm.t().expand(b, b) - 2 * torch.matmul(logits, logits.t())
    pairwise_euclid_distance = torch.clamp(pairwise_euclid_distance, min=0)
    expd = torch.clamp(torch.exp(-pairwise_euclid_distance / temperature) - torch.eye(b), min=0)
    loss = -(torch.log(STABILITY_EPS +
                       ((expd / (STABILITY_EPS + expd.sum(dim=1).expand(b, b))) * same_label_mask).sum(dim=1))).mean()

    return loss


def main():
    # Parsing command line parameters and necessary configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", help="JSON file configuring the program (layers, size of embeddings, ...)",
                        required=True, dest="conf_file_path")
    parser.add_argument("--ind-cache-manager", help="CacheManager file for individuals to reconcile", required=True,
                        dest="ind_cache_manager_file_path")
    parser.add_argument("--folds", help="File containing the similarity folds", required=True,
                        dest="similarity_folds_file_path")
    parser.add_argument("--graph", help="File path to the DGL graph to use", required=True)
    parser.add_argument("--model", help="Model to learn based on models defined in the configuration file", type=int,
                        required=True)
    parser.add_argument("--similarity-clusters", help="Similarity clusters computed by similarity_analysis.py",
                        required=True, dest="similarity_clusters_file_path")
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

    # Loading CacheManager for individuals to reconcile
    logger.info("Loading CacheManager for individuals to reconcile")
    ind_cache_manager = CacheManager()
    ind_cache_manager.load_from_csv(args.ind_cache_manager_file_path)

    # Loading folds
    folds = pickle.load(open(args.similarity_folds_file_path, "rb"))
    if len(folds) < 3:
        logger.critical("At least 3 folds needed")
        exit(-1)

    # Loading similarity clusters
    logger.info("Loading similarity clusters")
    similarity_clusters = pickle.load(open(args.similarity_clusters_file_path, "rb"))
    node_labels = from_similarity_clusters_to_labels(similarity_clusters[args.model], ind_cache_manager.get_size())
    node_to_cluster = from_similarity_clusters_to_mapping(similarity_clusters[args.model])

    # Building graph
    logger.info("Loading DGL graph")
    g = dgl.data.utils.load_graphs(args.graph)[0][0]

    # Learning
    logger.info("Model: {} -- Graph: {} -- {} folds".format(args.model, args.graph, len(folds)))
    results = dict()

    for i in tqdm.tqdm(range(0, len(folds)), desc="Test fold"):
        logger.info("Test fold: {} -- Validation fold: {}".format(i, (i + 1) % len(folds)))

        # Prepare dicts for loss
        test_set = set(folds[i])
        test_dict = from_learning_set_to_dict(test_set, node_to_cluster)
        val_set = set(folds[(i + 1) % len(folds)])
        val_dict = from_learning_set_to_dict(val_set, node_to_cluster)
        train_set = set().union(*[set(f) for f in folds]) - test_set - val_set
        train_dict = from_learning_set_to_dict(train_set, node_to_cluster)

        # Create model
        model = RGCNTModel(
            len(g),
            configuration_parameters["hidden-dim"],
            configuration_parameters["output-dim"],
            g.edata["rel_type"].max().item() + 1,
            num_bases=configuration_parameters["bases"],
            num_hidden_layers=configuration_parameters["hidden-layers"],
            temperature=configuration_parameters["temperature-init"]
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configuration_parameters["learning-rate"],
            weight_decay=configuration_parameters["l2norm-coeff"]
        )

        # Training
        logger.info("Start training...")
        epoch = 0

        early_stop = False
        early_stop_counter = 0
        val_loss_min = numpy.Inf

        model.train()

        logits_history = []
        train_loss_history = []
        val_loss_history = []
        test_loss_history = []
        temperature_history = []

        while epoch < configuration_parameters["epochs"] and not early_stop:
            # Forward pass
            logger.info("Forward pass")
            optimizer.zero_grad()
            logits = model.forward(g)

            # Train loss computation
            logger.info("Train loss computation")
            indices = list(train_dict.keys())
            indices.sort()
            train_loss = soft_nearest_neighbor_loss(
                logits[indices, :],
                torch.Tensor(node_labels)[indices, :],
                model.get_temperature()
            )

            # Backward pass
            logger.info("Backward pass")
            train_loss.backward()
            optimizer.step()

            # Validation loss computation
            logger.info("Validation loss computation")
            indices = list(val_dict.keys())
            indices.sort()
            val_loss = soft_nearest_neighbor_loss(
                logits[indices, :],
                torch.Tensor(node_labels)[indices, :],
                model.get_temperature()
            )

            # Test loss computation
            logger.info("Test loss computation")
            indices = list(test_dict.keys())
            indices.sort()
            test_loss = soft_nearest_neighbor_loss(
                logits[indices, :],
                torch.Tensor(node_labels)[indices, :],
                model.get_temperature()
            )

            # Saving histories
            logits_history.append(logits.detach().numpy()[0:ind_cache_manager.get_size(), :])
            train_loss_history.append(train_loss.item())
            val_loss_history.append(val_loss.item())
            test_loss_history.append(test_loss.item())
            temperature_history.append(model.get_temperature().item())

            # Display
            logger.info("T{}|V{}|E{:04d}|Train L: {:.10f} | Val L: {:.10f} | Test L: {:.10f} | Temp: {:.10f}".format(
                i,
                (i + 1) % len(folds),
                epoch,
                train_loss.item(),
                val_loss.item(),
                test_loss.item(),
                model.get_temperature().item()
            ))

            if any(torch.isnan(l) for l in [train_loss, val_loss, test_loss]) or \
                    any(torch.isinf(l) for l in [train_loss, val_loss, test_loss]):
                logger.critical("Loss NaN or Inf")
                exit(-1)

            # Test early stopping
            if val_loss.item() > val_loss_min - configuration_parameters["delta"]:
                early_stop_counter += 1

                if early_stop_counter >= configuration_parameters["patience"]:
                    logger.info("Early stopping")
                    early_stop = True

                    # Stop history at last ``checkpoint''
                    logits_history = logits_history[:epoch-early_stop_counter+1]
                    train_loss_history = train_loss_history[:epoch-early_stop_counter+1]
                    val_loss_history = val_loss_history[:epoch-early_stop_counter+1]
                    test_loss_history = test_loss_history[:epoch-early_stop_counter+1]

            else:
                early_stop_counter = 0
                val_loss_min = val_loss.item()

            epoch += 1

        results[i] = {
            "logits_history": logits_history,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "test_loss_history": test_loss_history,
            "temperature_history": temperature_history,
            "model": model.state_dict()
        }

    # Saving results
    output_dir = args.output_dir
    if output_dir[-1] != "/":
        output_dir += "/"

    logger.info("Saving results")
    pickle.dump(results, open(output_dir + "results", "wb"))


if __name__ == '__main__':
    main()
