# Coin Toss Problem:
# Estimate the head probabilities of the two biased coins A and B.
# Given the observed data of five sets of heads and tails, and each set is made only by either A or B coin,
# which is an unobserved variable:
# Set#1 {HTTTHHTHTH}, 5 heads
# Set#2 {HHHHTHHHHH}, 9 heads
# Set#3 {HTHHHHHTHH}, 8 heads
# Set#4 {HTHTTTHHTT}, 4 heads
# Set#5 {THHHTHHHTH}, 7 heads
#
# Use EM algorithm to estimate the parameter theta_A and theta_B, which are the probabilities of head on A and B.
#
# E-M algorithm:
# 1. Initialize the parameters theta_A and theta_B;
# 2. E-step: Use estimated parameters to compute expected heads distributions: heads_A and heads_B
# 3. M-step: maximize the log-likelihood by determining parameters using current heads distribution.
# 4. Repeat step 2 and 3 until convergence.
#############

import numpy as np
import matplotlib.pyplot as plt

# define constants
TOTAL_TOSS = 10
TOLERANCE = 1e-8

# set up observed data
heads = np.array([5,9,8,4,7], dtype=np.int)
tails = TOTAL_TOSS - heads
data = zip(heads, tails)

# compute probability of k heads out of total toss for a coin
def estimated_coin_probability(heads_prob_est, heads, tails):
    p, h, t = heads_prob_est, heads, tails
    if (heads_prob_est > 1.0):
        raise Exception("Probability should be within 0 to 1.")
    return (p**h)*((1-p)**t)


def main():
    # init parameters
    theta_A, theta_B = 0.5, 0.6 # head probability of coin A and B
    heads_A, heads_B = np.array([0.0]*len(data)), np.array([0.0]*len(data))
    tails_A, tails_B = np.array([0.0]*len(data)), np.array([0.0]*len(data))
    record_theta_A, record_theta_B = [theta_A], [theta_B] # keep record of parameter estimates

    # Expectation-Maximization algorithm
    iter_count = 0
    while (True):
        iter_count += 1
        prev_theta_A, prev_theta_B = theta_A, theta_B
        # E-step: use parameter to compute probability distribution
        index = 0
        for ht in data:
            prob_A = estimated_coin_probability(theta_A, ht[0], ht[1])
            prob_B = estimated_coin_probability(theta_B, ht[0], ht[1])
            tot_prob = prob_A + prob_B
            prob_A, prob_B = prob_A / tot_prob, prob_B / tot_prob  # normalize the data probability of A and B
            heads_A[index], heads_B[index] = prob_A * ht[0], prob_B * ht[0]  # heads distribution
            tails_A[index], tails_B[index] = prob_A * ht[1], prob_B * ht[1]  # tails distribution
            index += 1
        # M-step: estimate parameters
        theta_A = sum(heads_A) / (sum(heads_A) + sum(tails_A))
        theta_B = sum(heads_B) / (sum(heads_B) + sum(tails_B))
        record_theta_A.append(theta_A)
        record_theta_B.append(theta_B)
        # check convergence
        delta_A = (theta_A - prev_theta_A) / prev_theta_A
        delta_B = (theta_B - prev_theta_B) / prev_theta_B
        if (delta_A < TOLERANCE and delta_B < TOLERANCE):
            break
        else:
            continue

    # print results
    print "\t thetaA \t thetaB"
    print "\t {:.3f} \t {:.3f}".format(theta_A, theta_B)

    # plot the EM process
    fig = plt.figure(figsize=(12,12))
    fig.clf()
    ax = fig.gca()
    ax.plot(range(iter_count+1), record_theta_A, 'r--')
    ax.plot(range(iter_count + 1), record_theta_B, 'b--')
    #ax.set_yscale('log')
    #ax.set_ylim([0, 1.0])
    plt.show()


if __name__=="__main__":
    main()
