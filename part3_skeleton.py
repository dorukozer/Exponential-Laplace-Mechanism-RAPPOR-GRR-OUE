import math, random
import matplotlib.pyplot as plt
import numpy as np

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    p = np.exp(epsilon)/(np.exp(epsilon) + len(DOMAIN)-1)
    if random.random() < p:
        return int(val)
    else:
        rand =np.random.randint(0,len(DOMAIN)) +1
        while rand == val:
            rand =np.random.randint(0,len(DOMAIN))+1
        return int(rand)

    pass


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    p = np.exp(epsilon)/(np.exp(epsilon) + len(DOMAIN)-1)
    q = (1-p)/(len(DOMAIN)-1)

    unique, counts = np.unique(perturbed_values, return_counts=True)
    zipped=dict(zip(unique, counts))

    hist= np.zeros(len(DOMAIN))
    for val in np.array(DOMAIN):
        Iv=zipped[val]
        hist[val-1] = (Iv-(len(perturbed_values)*q))/(p-q)
    return hist.astype(int)
    pass


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    from  part2_skeleton import calculate_average_error
    pertube = np.zeros(len(dataset))
    for idx,val in enumerate(dataset):
        pertube[idx] = perturb_grr(val, epsilon)
    pertube=pertube.astype(int)
    unique, real_counts = np.unique(dataset, return_counts=True)
    estimated =estimate_grr(pertube, epsilon)
    return calculate_average_error(estimated,real_counts)






    pass


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    one_hot = np.zeros((np.array(DOMAIN).max() + 1))
    one_hot[val] = 1
    return(one_hot)
    pass


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = np.exp(epsilon/2)/(np.exp(epsilon/2) +1)
    encoded_val=np.copy(encoded_val)
    for idx,v in enumerate(encoded_val):
        if random.random() < p:
            encoded_val[idx] = v
        else:
            encoded_val[idx] = (v+1)%2
    return encoded_val
    pass


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    p = np.exp(epsilon/2)/(np.exp(epsilon/2) +1)
    q = 1/(np.exp(epsilon/2) +1)
    counts = np.sum(perturbed_values,axis=0)[1:]
    zipped=dict(zip(np.array(DOMAIN), counts))
    hist= np.zeros(len(DOMAIN))
    for val in np.array(DOMAIN):
        Iv=zipped[val]
        hist[val-1] = (Iv-(len(perturbed_values)*q))/(p-q)
    return hist.astype(int)
    pass
    pass


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    from  part2_skeleton import calculate_average_error

    pertube = np.zeros((len(dataset),np.array(DOMAIN).max()+1))
    for idx,val in enumerate(dataset):
        pertube[idx] = perturb_rappor(encode_rappor(val), epsilon)
    pertube=pertube.astype(int)

    unique, real_counts = np.unique(dataset, return_counts=True)

    estimated =estimate_rappor(pertube, epsilon)

    return calculate_average_error(estimated,real_counts)



    pass


# OUE

# TODO: Implement this function!
def encode_oue(val):
    one_hot = np.zeros((np.array(DOMAIN).max() + 1))
    one_hot[val] = 1
    return(one_hot)
    pass

# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    p = 1/(np.exp(epsilon) +1)
    p_1 = 1/2
    encoded_val=np.copy(encoded_val)
    for idx,v in enumerate(encoded_val):
        if v ==0:
            if random.random() < p:
                encoded_val[idx] = (v+1)%2

            else:
                encoded_val[idx] = v
        else:
            if random.random() < p_1:
                encoded_val[idx] = v
            else:
                encoded_val[idx] = (v+1)%2
    return encoded_val
    pass

    pass


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    p = 1/(np.exp(epsilon) +1)
    q =  np.exp(epsilon)/(np.exp(epsilon) +1)
    p_1 = 1/2
    counts = np.sum(perturbed_values,axis=0)[1:]
    zipped=dict(zip(np.array(DOMAIN), counts))
    hist= np.zeros(len(DOMAIN))
    for val in np.array(DOMAIN):
        Iv=zipped[val]
        hist[val-1] = (2*((np.exp(epsilon)+1)*Iv -len(perturbed_values)))  /(np.exp(epsilon)-1)
    return hist.astype(int)
    pass


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    from  part2_skeleton import calculate_average_error
    pertube = np.zeros((len(dataset),np.array(DOMAIN).max()+1))
    for idx,val in enumerate(dataset):
        pertube[idx] = perturb_oue(encode_oue(val), epsilon)
    pertube=pertube.astype(int)
    unique, real_counts = np.unique(dataset, return_counts=True)
    estimated =estimate_oue(pertube, epsilon)
    return calculate_average_error(estimated,real_counts)

    pass


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")
    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
   
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

