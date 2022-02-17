import csv
from knn_algorithm import knn_predict, normalize, transpose


# Open dataset from file
file_name = "KNNAlgorithmDataset.csv"
with open(file_name) as csv_file:
    data = list(csv.reader(csv_file))


# Normalization of data -> [0, 1] (specific to dataset format)
#_____________________________
normalized_data = []
for row in range(2, len(data[0])-1): # assuming all columns have the same row length, starting at range(2) since row 1 & 2 are labels
    column = [float(column[row]) for column in data[1:]]
    normalized_data.append([normalize(x, min(column), max(column)) for x in column])

normalized_data = transpose(normalized_data) # transposing matrix col -> rows
#_____________________________


# Splitting data into training and validating
#_____________________________
training_data_size = 400 # less than total data size
k_nearest = 23 # arbitrary value, must be odd to account for class ties

training_data = normalized_data[:training_data_size]  # start at index 1 since not including labels from row 1->2
labels = [column[1] for column in data[1:]] # extract label from from dataset
#_____________________________


# Checking accuracy of model
#_____________________________
num_correct = 0
for x in range(training_data_size, len(normalized_data)): # using remaining data as validation
    prediction = knn_predict(normalized_data[x], training_data, labels, k_nearest)
    actual = labels[x]
    if prediction == actual:
        num_correct += 1

percent_correct = num_correct/((len(data)-1)-training_data_size)*100
print(f"percent correct={round(percent_correct, 1)}%")
#_____________________________
