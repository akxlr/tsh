import numpy as np
import numpy.linalg as npla
import numpy.random as npr
import math

alphabet = "abcdefghijklmnopqrstuvwxyz"

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

def gd(v, iterations):
    """Gradient descent starting at v=[n_a, n_b, ..., n_z] for a given number of iterations, return final v"""
    for iteration in range(iterations):
        e = error(v)
        new_v = v.copy()
        # Check each of the 26 directions, move in one of them (or stay put if no direction reduces error)
        for i in range(len(alphabet)):
            v[i] += 1
            if error(v) < e:
                e = error(v)
                new_v = v.copy()
            v[i] -= 2
            if error(v) < e:
                e = error(v)
                new_v = v.copy()
            v[i] += 1
        v = new_v
    return v

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
    print("Start: problem size 2^%s" % math.ceil(math.log(domain_size(domain), 2)))
    for iteration in range(iterations):
        for i in range(len(alphabet)):
            i_counts = [letter_counts[domain[j], i] for j in range(len(alphabet))]
            domain[i] = slice(sum(np.amin(x) for x in i_counts) + b[i], sum(np.amax(x) for x in i_counts) + 1 + b[i])
        print("Iteration %s: problem size 2^%s" % (iteration, math.ceil(math.log(domain_size(domain), 2))))
        print(", ".join(
            "%s %s %s" % (alphabet[i], domain[i].start, domain[i].stop) if domain[i].stop != domain[i].start + 1
            else "%s=%s" % (alphabet[i], domain[i].start) for i in range(len(alphabet))))
    return domain

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

def solve(iterations):
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
