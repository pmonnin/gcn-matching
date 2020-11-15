import numpy
import tqdm


def from_similarity_clusters_to_labels(similarity_clusters, individuals_number):
    labels = numpy.zeros((individuals_number, 1), dtype=int)

    for i, c in enumerate(similarity_clusters):
        for ind in c:
            labels[ind] = i

    return labels


def size_constrained_clusters_to_labels(similarity_clusters, individuals_number, min_size):
    labels = -numpy.ones((individuals_number,), dtype=int)

    i = 0
    for c in similarity_clusters:
        if len(c) >= min_size:
            for ind in c:
                labels[ind] = i
            i += 1

    return labels


def from_similarity_clusters_to_colors(similarity_clusters, individuals_number, max_color, max_size):
    colors = numpy.zeros(individuals_number, dtype=int)
    color_list = list(range(1, max_color+1))

    for i, c in enumerate(sorted(similarity_clusters, reverse=True)):
        if max_size <= 0 or len(c) < max_size:
            if len(color_list) == 0:
                color_list.append(0)

            if i % 2 == 0:
                color_i = color_list[0]
            else:
                color_i = color_list[-1]
            color_list.remove(color_i)

            for ind in c:
                colors[ind] = color_i

    return colors


def from_similarity_clusters_to_mapping(similarity_clusters):
    mapping_node_to_cluster = dict()

    for c in similarity_clusters:
        for i in c:
            mapping_node_to_cluster[i] = c

    return mapping_node_to_cluster


def from_learning_set_to_dict(learning_set, node_to_cluster):
    learning_dict = dict()

    for i in tqdm.tqdm(learning_set):
        if len(node_to_cluster[i] & learning_set) > 1:
            learning_dict[i] = set(node_to_cluster[i] & learning_set) - {i}

    return learning_dict
