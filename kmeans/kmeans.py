import numpy as np

class KMeans:
    def __init__(self, k, distance_function, medoids = True):
        self.k = k
        self.distance_function = distance_function
        self.data = None
        self.label_data = None
        self.data_shape = None
        self.data_length = None
        self.centroids = []
        self.clusters = {i:}[] for i in range(self.k)}
        self.clusters_idx = {i:[] for i in range(self.k)}
        self.assignment = None
        self.labels = None

    def _load_data(data, label_data):
        self.data = data
        self.label_data = label_data
        self.data_shape = data.shape[1:]
        self.data_length = data.shape[0]
        self.labels = [-1]*self.data_length

    def _initialization():
        self.centroids = [np.zeros(self.data_shape)]*self.k
        self.assignment = {i:-1, for i in range(self.data_length)}
        for i in range(k):
            rand_index = np.random.randint(0, self.data_length, 1)
            self.centroids[i] = self.data[rand_index]

    def _calculate_medoid(data):
        dist = np.zeros(data.shape[0], data.shape[0])
        for i in data.shape[0]:
            for j in data.shape[0]:
                dist[i,j] = self.distance_function(data_i,data_j)
        dist = np.sum(dist, axis = 1)
        return np.where(dist == np.amin(dist))


    def fit(data, label_data, verbos = True, print_freq = 10):
        self._load_data(data, label_data)
        self._initialization()

        iter = 0
        change_count = -1

        while change_count != 0:
            if verbos and iter%print_freq == 0:
                print("Iteration: ", iter)
                print(" Centroids: ", new_centroids)
                print("Last Iteration Changed ", change_count, " Centroids.")
            change_count = 0
            iter += 1

            self.clusters = {i:}[] for i in range(self.k)}

            for i in range(self.data_length):
                dist_min = float("inf")
                cluster_assignemnt = -1
                for c in range(self.k):
                    dist_c = self.distance_function(data[i], self.centroids[c])
                    if dist_c < dist_min:
                        dist_min = dist_c
                        cluster_assignemnt = c
                self.clusters[cluster_assignemnt].append(data[i])
                self.clusters_idx[cluster_assignemnt].append(i)
                self.assignment[i] = cluster_assignemnt

            for j in range(self.k):
                if not self.medoids:
                    new_centroid = np.mean(np.array(self.clusters[j]), axis = 0)
                else:
                    new_centroid = self._caluclate_medoid(self.clusters[j])
                if self.distance_function(new_centroid, self.centroids[j]) > 0:
                    change_count += 1
                self.centroids[i] = new_centroid

    def predict(data):
        assignment = []
        for data_i in data:
            dist_min = float("inf")
            cluster_assignemnt = -1
            for c in range(self.k):
                dist_c = self.distance_function(data_i, self.centroids[c])
                if dist_c < dist_min:
                    dist_min = dist_c
                    cluster_assignemnt = c
            assignment.append(cluster_assignemnt)
        return assignment

    def fit_predict(train_data, test_data, verbos = True, print_freq = 10):
        self.fit(train_data, verbos, print_freq)
        return(test_data)

    def generate_label_for_clusters():
        for j in range(self.k):
            values,counts = np.unique(self.label_data[self.clusters_idx[j]], return_counts=True)
            idx = np.argmax(counts)
            self.labels[j] = values[idx]

    def predict_labels(test_data):
        assignments = self.predict(test_data)
        return [self.labels[assignment] for assignment in assignments]
