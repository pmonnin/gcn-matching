import csv

__author__ = "Pierre Monnin"


class CacheManager:
    def __init__(self):
        self._cache = {}
        self._inverse_cache = []

    def get_element_index(self, element):
        if element not in self._cache:
            self._cache[element] = len(self._cache)
            self._inverse_cache.append(element)

        return self._cache[element]

    def get_element_from_index(self, index):
        if index > len(self._inverse_cache):
            return ""

        return self._inverse_cache[index]

    def is_element_in_cache(self, element):
        return element in self._cache

    def get_size(self):
        return len(self._cache)

    def save_to_csv(self, file):
        with open(file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            for k, i in self._cache.items():
                writer.writerow([k, i])

    def load_from_csv(self, file):
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                self._cache[row[0]] = int(row[1])

        self._inverse_cache = [""] * len(self._cache)
        for k, v in self._cache.items():
            self._inverse_cache[int(v)] = k

    def __str__(self):
        retval = "-- CacheManager --\n"
        for uri, i in self._cache.items():
            retval += uri + " <=> " + str(i) + "\n"
        retval += "--"
        return retval
