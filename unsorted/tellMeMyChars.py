import sys

if __name__ == "__main__":
    res = input("Put in your characters:")
    for c in res:
        if c == "\n":
            print("\\n", end="")
        elif c == "\r":
            print("\\r", end="")
        else:
            print(c, end="")
