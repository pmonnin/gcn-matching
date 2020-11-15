import json
import logging
import socket

import requests
import requests.auth
import tqdm

__author__ = "Pierre Monnin"


class ServerManager:
    def __init__(self, configuration_parameters, max_rows):
        # Server address and GET query parameters
        self.server_address = configuration_parameters["server-address"]
        self.json_conf_attribute = configuration_parameters["url-json-conf-attribute"]
        self.json_conf_value = configuration_parameters["url-json-conf-value"]
        self.default_graph_attribute = configuration_parameters["url-default-graph-attribute"]
        self.default_graph_value = configuration_parameters["url-default-graph-value"]
        self.query_attribute = configuration_parameters["url-query-attribute"]
        socket.setdefaulttimeout(configuration_parameters["timeout"])

        # Username / password
        self.username = configuration_parameters["username"]
        self.password = configuration_parameters["password"]

        # Max rows for SPARQL result sets
        self.max_rows = max_rows

        # Prefixes for SPARQL queries
        self.prefixes = ""
        for prefix, uri in configuration_parameters["prefixes"].items():
            self.prefixes += "PREFIX %s:<%s> \n" % (prefix, uri)

        # Logging
        self._logger = logging.getLogger()

    def query_server(self, query):
        done = False

        content = {}

        while not done:
            done = True

            query_parameters = {
                self.json_conf_attribute: self.json_conf_value,
                self.default_graph_attribute: self.default_graph_value,
                self.query_attribute: self.prefixes + query
            }

            if self.username != "" and self.password != "":
                content = requests.get(self.server_address, query_parameters,
                                       auth=requests.auth.HTTPBasicAuth(self.username, self.password))

            else:
                content = requests.get(self.server_address, query_parameters)

            if content.status_code == 404:
                done = False
                self._logger.critical("404 error. New try...")

            elif content.status_code != 200:
                self._logger.critical(content.content)
                exit(-1)

        return json.loads(content.text)

    def query_count_elements(self, where_clause):
        results_json = self.query_server("select count(distinct ?e) as ?count where { " + where_clause + " }")
        return int(results_json["results"]["bindings"][0]["count"]["value"])

    def query_count_two_elements(self, where_clause):
        results_json = self.query_server("select count(*) as ?count where { select distinct ?e1 ?e2 where {" +
                                         where_clause + " } }")
        return int(results_json["results"]["bindings"][0]["count"]["value"])

    def query_elements(self, where_clause, verbose=False):
        ret_val = []
        elements_count = self.query_count_elements(where_clause)

        if verbose and elements_count != 0:
            pbar = tqdm.tqdm(total=elements_count)

        while len(ret_val) != elements_count:
            ret_val = []
            offset = 0

            while offset <= elements_count:
                results_json = self.query_server("select distinct ?e where { " + where_clause + " } LIMIT "
                                                 + str(self.max_rows) + " OFFSET " + str(offset))

                for result in results_json["results"]["bindings"]:
                    ret_val.append(str(result["e"]["value"]))

                    if verbose and elements_count != 0:
                        pbar.update(1)

                offset += self.max_rows

            if len(ret_val) != elements_count:
                self._logger.error("Number of elements different from count, retry...")

                if verbose and elements_count != 0:
                    pbar.close()
                    pbar = tqdm.tqdm(total=elements_count)

        if verbose and elements_count != 0:
            pbar.close()

        return ret_val

    def query_two_elements(self, where_clause, verbose=False):
        ret_val = []
        elements_count = self.query_count_two_elements(where_clause)

        if verbose and elements_count != 0:
            pbar = tqdm.tqdm(total=elements_count)

        while len(ret_val) != elements_count:
            ret_val = []
            offset = 0

            while offset <= elements_count:
                results_json = self.query_server("select distinct ?e1 ?e2 where { " + where_clause + " } LIMIT "
                                                 + str(self.max_rows) + " OFFSET " + str(offset))

                for result in results_json["results"]["bindings"]:
                    ret_val.append((str(result["e1"]["value"]), str(result["e2"]["value"])))

                    if verbose and elements_count != 0:
                        pbar.update(1)

                offset += self.max_rows

            if len(ret_val) != elements_count:
                self._logger.error("Number of elements different from count, retry...")

                if verbose and elements_count != 0:
                    pbar.close()
                    pbar = tqdm.tqdm(total=elements_count)

        if verbose and elements_count != 0:
            pbar.close()

        return ret_val
