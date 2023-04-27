import sys

def isEven(n):
    return not (n % 2)
def hailstone(n):
    assert not isinstance(n, float)
    assert n > 0
    if isEven(n):
        return n//2
    else:
        return 3*n + 1

def main():
    #if len(sys.argv) < 2:
        #print("Use: python3 hailstone.py <n>")
        #return

    #n = int(sys.argv[1])

    for n in range(100000000, 1000000000):
        print(n, end='\r')
        path = [n]
        while path[-1] != 1:
            path.append(hailstone(path[-1]))

        if len(path) > 10000:
            print("")
            print("Path length: " +str(len(path)))


    

if __name__ == "__main__":
    main()
