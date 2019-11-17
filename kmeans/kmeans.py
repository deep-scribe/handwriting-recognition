import numpy as np

class KMeans:
    def __init__(self, k, distance_function, medoids = True):
        self.k = k
        self.distance_function = distance_function
        self.medoids = medoids
        self.data = None
        self.label_data = None
        self.data_shape = None
        self.data_length = None
        self.centroids = []
        self.clusters = {i:[] for i in range(self.k)}
        self.clusters_idx = {i:[] for i in range(self.k)}
        self.assignment = None
        self.labels = None

    def _load_data(self, data, label_data):
        self.data = data
        self.label_data = label_data
        self.data_shape = data.shape[1:]
        self.data_length = data.shape[0]
        self.labels = [-1]*self.data_length

    def _initialization(self):
        self.centroids = [np.zeros(self.data_shape)]*self.k
        self.assignment = {i:-1 for i in range(self.data_length)}
        for i in range(self.k):
            rand_index = np.random.randint(0, self.data_length, 1)
            self.centroids[i] = self.data[rand_index].reshape(self.data_shape)

    def _calculate_medoid(self, data):
        dist = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                dist[i,j] = self.distance_function(data[i],data[j])
        dist = np.sum(dist, axis = 1)
        return np.where(dist == np.amin(dist))


    def fit(self, data, label_data, verbos = True, print_freq = 10):
        self._load_data(data, label_data)
        self._initialization()

        iter = 0
        change_count = -1

        while change_count != 0:
            if verbos and iter%print_freq == 0:
                print("Iteration: ", iter)
                # print(" Centroids: ", self.centroids)
                print("Last Iteration Changed ", change_count, " Centroids.")
            change_count = 0
            iter += 1

            self.clusters = {i:[] for i in range(self.k)}

            for i in range(self.data_length):
                dist_min = float("inf")
                cluster_assignemnt = -1
                for c in range(self.k):
                    dist_c = self.distance_function(data[i], self.centroids[c])
                    # print(dist_c)
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
                    try:
                        new_centroid_index = int(self._calculate_medoid(self.clusters[j])[0])
                        new_centroid = self.clusters[j][new_centroid_index]
                    except:
                        new_centroid = self.centroids[j]
                if self.distance_function(new_centroid, self.centroids[j]) > 0:
                    change_count += 1
                self.centroids[j] = new_centroid
        self.generate_label_for_clusters()

    def predict(self, data):
        assignment = []
        for i in range(data.shape[0]):
            dist_min = float("inf")
            cluster_assignemnt = -1
            for c in range(self.k):
                # print(data[i].shape, self.centroids[c].shape)
                dist_c = self.distance_function(data[i], self.centroids[c])
                if dist_c < dist_min:
                    dist_min = dist_c
                    cluster_assignemnt = c
            assignment.append(cluster_assignemnt)
        return assignment

    def fit_predict(self, train_data, test_data, verbos = True, print_freq = 10):
        self.fit(train_data, verbos, print_freq)
        return(test_data)

    def generate_label_for_clusters(self):
        for j in range(self.k):
            # print(j, len(self.clusters[j]))
            values,counts = np.unique(self.label_data[self.clusters_idx[j]], return_counts=True)
            # print(values,counts)
            try:
                idx = np.argmax(counts)
                self.labels[j] = values[idx]
            except:
                self.labels[j] = "-1"

    def predict_labels(self, test_data):
        assignments = self.predict(test_data)
        return [self.labels[assignment] for assignment in assignments]
