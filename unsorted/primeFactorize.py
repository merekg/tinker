import sys
import matplotlib.pyplot as plt
import numpy as np

def isPrime(n):
    for i in range(2,n):
        if n % i == 0:
            return False
    return True

def primeFactor(N,factors):
    if(N == 1):
        return factors
    for prime in range(2,N+1):
        if N % prime == 0 and isPrime(prime):
            factors.append(prime)
            return primeFactor(int(N//prime), factors)

sys.setrecursionlimit(2000)
#N = int(sys.argv[1])
#factors = []

#factored = primeFactor(N,factors)
#print(factored)

NN = range(1,1000000)
nPrimeFactors = []
for n in NN:
    factors = []
    nPrimeFactors.append(len(primeFactor(n, factors)))


#plt.scatter(NN, nPrimeFactors)
#m,b = np.polyfit(NN, nPrimeFactors, 1)
#plt.plot(NN, m*NN+b)
plt.hist(nPrimeFactors)
plt.show()
