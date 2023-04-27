import sys

fp = sys.argv[1]
N = int(sys.argv[2])

count = 0
for line in open(fp):
    if(count % N is not 0):
        print(line, end='')
    count += 1

