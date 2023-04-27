import sys
import numpy as np
import math
import time

def isPrime(n):
    for i in range(2, math.floor(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def multiples(n,M):
    multiples = [n*2]
    while multiples[-1] < M:
        multiples.append(n * (len(multiples)+2))
    return multiples

def naivePrimes(N):
    primes = []
    for n in range(N):
        if isPrime(n):
            primes.append(n)

def sieve(N):
    allNumbers = range(2,N)
    primes = []
    composites = set()
    for n in range(2, N):
        if n not in composites and isPrime(n):
            primes.append(n)
            composites.update(multiples(n,N))

# parse inputs
N = int(sys.argv[1])

start = time.time()
for i in range(10000):
    sieve(N)
end = time.time()
print(end - start)

start = time.time()
for i in range(10000):
    naivePrimes(N)
end = time.time()
print(end - start)
