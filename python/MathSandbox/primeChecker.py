import sys

def is_prime(n):
    assert isinstance(n,int)
    for i in range(2,n//2 + 1):
        if n % i is 0:
            return False
    return True

assert sys.argv[1].isnumeric()
primes = 0
N = int(sys.argv[1])
for n in range(int(sys.argv[1])):
    if is_prime(n):
        print(n)
        primes +=1
print("Found", primes, "primes")
