import numpy as np
import numpy.linalg as npla
import numpy.random as npr
from numpy.testing import assert_allclose
import math
from matplotlib import pyplot as plt
from scipy.stats import rv_discrete

# Tolerances for assert_allclose (floating point equality assertions)
_RTOL = 1e-03
_ATOL = 1e-08

alphabet = "abcdefghijklmnopqrstuvwxyz"
np.set_printoptions(precision=2)

def num2str(n):
    assert 0 <= n <= 99
    ones_teens = "no,one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,seventeen,eighteen,nineteen".split(",")
    tens = "twenty,thirty,forty,fifty,sixty,seventy,eighty,ninety".split(",")
    if n < 20:
        return ones_teens[n]
    else:
        s = tens[(n // 10) - 2]
        if n % 10 == 0:
            return s
        else:
            return "%s-%s" % (s, ones_teens[n % 10])
    

def letter_counts_str(s):
    return np.array([s.count(letter) for letter in alphabet])

letter_counts = np.array([letter_counts_str(num2str(n) + ('' if n==1 else 's')) for n in range(100)])
b = letter_counts_str("this sentence has" + alphabet + "and")

z = np.zeros(len(alphabet), dtype='int')

def create_sentence(u):
    """Not used in algorithm at all - just for printing/debugging"""
    s = "This sentence has "
    for i in range(len(alphabet)):
        s += num2str(u[i]) + " " + alphabet[i] + ("" if u[i]==1 else "s")
        if i == len(alphabet) - 2:
            s += " and "
        elif i == len(alphabet) - 1:
            s += "."
        else:
            s += ", "
    return s

def f(u):
    """Return total number of each letter if sentence was constructed using assignments in u. Goal is then to find fixed point of f (where f(u) == u)."""
    # Extract the relevant rows from letter_counts to give the letter counts for each entry of u, then sum the columns
    return letter_counts[u, :].sum(axis=0) + b

def error(u):
    return npla.norm(f(u)-u)

def gd(v):
    """Gradient descent starting at v=[n_a, n_b, ..., n_z]"""
    while(True):
        e = error(v)
        # Check each of the 26 directions, move in one of them (or return if no direction reduces error)
        changed = False
        for i in range(len(alphabet)):
            v[i] += 1
            if error(v) < e:
                e = error(v)
                new_v = v.copy()
                changed = True
            v[i] -= 2
            if error(v) < e:
                e = error(v)
                new_v = v.copy()
                changed = True
            v[i] += 1
        if not changed:
            return v
        else:
            v = new_v

def domain_size(domain):
    """Return cardinality of domain given a set of restrictions, i.e. number of iterations required to brute
       force the problem. `domain` is a list of python slice objects."""
    # np.product does not handle large numbers, we need normal python arithmetic
    result = 1
    for s in domain:
        result *= (int(s.stop) - int(s.start))
    return result

def minmax(iterations):
    """Barney's method: reduce the domain of possible values for each letter. Start by assuming all values in [0,99], then iteratively
       reduce the domain of each letter using the domain of all other letters. Repeat for the given number of iterations.
       Results in reduction of problem size from 2^173 to 2^73, converges after about 4 iterations."""
    domain = [slice(0, 100) for letter in alphabet]
#     print("Start: problem size 2^%s" % math.ceil(math.log(domain_size(domain), 2)))
    for iteration in range(iterations):
        for i in range(len(alphabet)):
            i_counts = [letter_counts[domain[j], i] for j in range(len(alphabet))]
            domain[i] = slice(sum(np.amin(x) for x in i_counts) + b[i], sum(np.amax(x) for x in i_counts) + 1 + b[i])
#         print("Iteration %s: problem size 2^%s" % (iteration, math.ceil(math.log(domain_size(domain), 2))))
    print(", ".join(
            "%s %s %s" % (alphabet[i], domain[i].start, domain[i].stop) if domain[i].stop != domain[i].start + 1
            else "%s=%s" % (alphabet[i], domain[i].start) for i in range(len(alphabet))))
    return domain

def graph_p(p, letter_idx):
    pass

def letter_prob(p, letter):
    """Input: probability of each value
       Output: probability of each letter count"""
    assert_allclose(np.sum(p), 1, rtol=_RTOL, atol=_ATOL)
    buckets = np.zeros(100, dtype='float64')
    for n in range(p.size):
        buckets[letter_counts[n, letter]] += p[n]
    assert_allclose(np.sum(buckets), 1, rtol=_RTOL, atol=_ATOL)
    return buckets

def prob_sum(p1, p2):
    """Probability distribution of sum of two random variables. See http://www.statlect.com/sumdst1.htm."""
    # We assume sum will never be greater than 99 so p will always have enough elements - this is
    # equivalent to assuming no values in solution are greater than 99
    assert_allclose(np.sum(p1), 1, rtol=_RTOL, atol=_ATOL)
    assert_allclose(np.sum(p2), 1, rtol=_RTOL, atol=_ATOL)
    p = np.zeros_like(p1, dtype='float64')
    for n in range(p.size):
        p[n] = min(1, sum(p1[x] * p2[n - x] for x in range(0, n + 1)))
    assert_allclose(np.sum(p), 1, rtol=_RTOL, atol=_ATOL)
    return p

def prob(iterations):
    domain = minmax(10)
    # Initialise each letter to uniform distribution
    p_mat = np.zeros_like(letter_counts, dtype='float64')
    for i in range(len(alphabet)):
        p_mat[domain[i], i] = 1/(domain[i].stop - domain[i].start)

    for iteration in range(iterations):
        print("Iteration %s" % iteration)
        new_p_mat = np.zeros_like(p_mat, dtype='float64')
        for i in range(len(alphabet)):
            # Calculate new distribution for letter i
            # Start with [1, 0, ..., 0] (length 100), then repeatedly update using rule for sum of random variables.
            # Result is that each entry of p_total is the probability of letter i being that number based on current
            # distribution stored in p
            p_total = np.zeros(letter_counts.shape[0], dtype='float64')
            p_total[0] = 1 # [1, 0, 0, ..., 0] is the identity element for prob_sum
            for j in range(len(alphabet)):
                p_total = prob_sum(p_total, letter_prob(p_mat[:, j], i))

            # np.roll shifts array elements right a certain number of places - this is necessary for the bias b
            new_p_mat[:, i] = np.roll(p_total, b[i])
        p_mat = new_p_mat
        plt.bar(range(100), p_mat[:, alphabet.index('s')])
        plt.savefig('s%s.png' % iteration)
        plt.clf()
        plt.bar(range(100), p_mat[:, alphabet.index('t')])
        plt.savefig('t%s.png' % iteration)
        plt.clf()
        plt.bar(range(100), p_mat[:, alphabet.index('f')])
        plt.savefig('f%s.png' % iteration)
        plt.clf()
    return p_mat




def iter(v, iterations):
    """Repeat f(f(...f(v))) for a given number of iterations, return best v seen"""
    e = error(v)
    best_v = v
    for iteration in range(iterations):
        if error(v) < e:
            e = error(v)
            best_v = v
        v = f(v)
    return best_v

def rand_start(domain):
    return np.array([npr.randint(s.start, s.stop) for s in domain])

# def sample(p, size):
#     bins = np.add.accumulate(p)
#     return np.array([np.digitize(npr.random_sample(size), bins[:, i]) for i in range(p.shape[1]))

def solve_2(iterations):
    """Draw random samples according to the distribution of values
       found when prob() converges, and hope one is a solution."""
    # Empirically prob() converges after ~10 iterations, should probably
    # make this concrete. Also, more than 10 iterations causes numerical assertion
    # errors, not sure why...

    # Output of prob(10) is cached on disk using np.save since it takes a while
    p_mat = np.load("prob_10.npy")
    # I can't find a way of sampling from a multivariate discrete distribution
    # in numpy directly, so we use 26 univariate samplers
    dist = [rv_discrete(values=(np.arange(p_mat.shape[0]), p_mat[:, i])) for i in range(len(alphabet))]
    batch_size = 100
    best = z
    e = error(z)
    for iteration in range(iterations):
        print("Iteration %s (batch size %s)" % (iteration, batch_size))
        samples = np.array([dist[i].rvs(size=batch_size) for i in range(len(alphabet))]).T
        print("  Generated %s samples, trying gd-iter-gd on all..." % batch_size)
        for v in samples:
            v = gd(iter(gd(v), 100))
            if error(v) < e:
                e = error(v)
                best = v
        print("  Best error so far %s" % e)
    return best


def solve_1(iterations):
    domain = minmax(10)
    best = z
    e = error(z)
    for iteration in range(iterations):
        v = rand_start(domain)
        v = iter(v, 5000)
        v = gd(v, 10)
        if error(v) < e:
            e = error(v)
            best = v
    print(error(best))
    return best
