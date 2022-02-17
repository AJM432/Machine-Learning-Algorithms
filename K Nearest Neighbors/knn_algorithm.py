# Modules used in KNN algorithm

# finds euclidian distance of n dimensional arrays
def euclid_distance(a, b):  # a and b are points with n dimensions
    return sum((elem[1]-elem[0])**2 for elem in zip(a, b))**(1/2)


# convert a value to a range [0, 1]
def normalize(value, value_min, value_max):
    return (value - value_min) / (value_max - value_min)


# finds n lowest numbers in array
def smallest_nums_array(array, n):
    array.sort()
    return array[:n]


# Switches array rows with columns
def transpose(array):
    return [[array[j][i] for j in range(len(array))] for i in range(len(array[0]))]


# data must be normalized by each feature min and max values
def knn_predict(test_point, training_data, training_labels, k_nearest):
    # Must compute distances of all data points from test point and find the nearest neighbors
    distances = []

    for index, item in enumerate(training_data):
        distances.append([euclid_distance(item, test_point), index]) # format: [distance, index from original list "for reference"]

    closest_points = smallest_nums_array(distances, k_nearest)
    
    # Computes the frequency of a class from closest points; highest class count determines test_point class prediction
    class_frequency = {}
    for point in closest_points:
        # Accessing second element point[1] for index from original data list
        if training_labels[point[1]] not in class_frequency:
            class_frequency[training_labels[point[1]]]=1
        else:
            class_frequency[training_labels[point[1]]] += 1
    return max(class_frequency, key=class_frequency.get) # get highest class value
