import statistics

def calculate_central_tendency_measures(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    mode = statistics.mode(data)
    return mean, median, mode

def calculate_measures_of_dispersion(data):
    variance = statistics.variance(data)
    std_dev = statistics.stdev(data)
    return variance, std_dev

data = [12, 15, 18, 20, 22, 25, 28, 30]

mean, median, mode = calculate_central_tendency_measures(data) 
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

variance, std_dev = calculate_measures_of_dispersion(data)
print("Variance:", variance)
print("Standard Deviation:", std_dev)