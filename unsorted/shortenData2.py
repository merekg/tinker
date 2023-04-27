import sys

f = open(sys.argv[1],'r')

i = 0
for line in f:
    i = i+1
    if i % 20 is 0:
        print(line, end="")
