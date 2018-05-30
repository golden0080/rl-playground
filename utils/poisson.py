import math


def poisson_possibility_split(lam, max_x):
    x = 0
    p = math.exp(-lam)
    s = 0
    distributions = []
    while x < max_x:
        s += p
        distributions.append(p)
        x += 1
        p *= float(lam) / float(x)

    distributions.append(1 - s)
    return distributions
