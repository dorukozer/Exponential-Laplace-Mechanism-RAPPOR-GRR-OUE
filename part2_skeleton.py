import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    df=(dataset)
    anime_map = np.zeros((len(df.columns),13))
    skip = 0

    for (columnName, columnData) in df.items():
        if skip !=0:

            if int(columnName) == int(chosen_anime_id):
                counts, bins = np.histogram(columnData, bins=[-1,0, 1, 2, 3,4,5,6,7,8,9,10,11])
                return counts.astype(int)
        skip=1

pass


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    sensitivity=2
    counts2 = np.copy(counts)
    for idx, x in enumerate(counts2):
        counts2[idx]+= np.random.laplace(loc=0.0, scale=sensitivity/epsilon, size=None)
    return counts2

    pass


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    dif=np.abs(actual_hist-noisy_hist)
    return np.mean(dif)
    pass


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    dif=np.abs(actual_hist-noisy_hist)**2
    return np.mean(dif)
    pass


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    mse = np.zeros(len(eps_values))
    avg = np.zeros(len(eps_values))
    for idx,eps in enumerate(eps_values):
        for i in range(40):
            dp_hist= get_dp_histogram(counts, eps)
            mse[idx] +=calculate_mean_squared_error(counts,dp_hist)
            avg[idx]+=calculate_average_error(counts,dp_hist)
    return avg/40, mse/40
    pass


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    all_hists= np.zeros((len(dataset.columns)-1,12))
    cols = list(dataset.columns)[1:]
    qfs =np.zeros((12))
    sens=1
    for idx,anime in enumerate(list(dataset.columns)[1:]):
            all_hists[idx] = get_histogram(dataset, chosen_anime_id=int(anime))
            qfs[idx] = all_hists[idx][11]
    p = [np.exp(epsilon * qf / (2 * sens)) for qf in qfs]
    p = p / np.linalg.norm(p, ord=1)

    return np.random.choice(cols, 1, p=p)[0]
    pass



def helper_actual(dataset):
    all_hists= np.zeros((len(dataset.columns)-1,12))
    cols = list(dataset.columns)[1:]
    for idx,anime in enumerate(list(dataset.columns)[1:]):
        all_hists[idx] = get_histogram(dataset, chosen_anime_id=int(anime))
    best = -1
    max = -1
    for idx,lap_hist in enumerate(all_hists):
        if lap_hist[11] >best:
            best = lap_hist[11]
            max = idx
    return cols[max]

# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    accur = np.zeros(len(eps_values))
    actually_most_rated=helper_actual(dataset)
    for idx,eps in enumerate(eps_values):
        for i in range(1000):
            the_most_rated= most_10rated_exponential(dataset, eps)
            if the_most_rated == actually_most_rated:
                accur[idx] +=1

    return accur/10
    pass


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])


    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()

