import random

def parse_command(command):
    # if the first char is a d, insert the implicit 1
    if command[0] is 'd':
        command = '1' + command

    c = command.split('d')
    if len(c) is not 2:
        raise Exception("Ill-formed command")
    for e in c:
        if not e.isnumeric():
            raise Exception("Ill-formed command")
    return [int(e) for e in c]

def main():
    random.seed()
    while(True):
        command = input(">")
        try:
            n,d = parse_command(command)
            s = 0
            for i in range(n):
                s += random.randint(1,d)
            print(s)
        except Exception as e:
            print("Ill-formed command")

if __name__ == "__main__":
    main()
