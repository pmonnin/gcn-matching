FROM python:3.7

RUN pip -q install requests tqdm dgl torch torchvision matplotlib scikit-learn umap-learn pynndescent

WORKDIR /gcn-matching

COPY src /gcn-matching/
