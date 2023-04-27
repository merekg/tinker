import sys
import os

if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert os.path.isfile(sys.argv[1])

    inpath = sys.argv[1]
    infile = open(inpath, 'r')
    lines = infile.readlines()

    trainer = "";
    for line in lines:
        arr = line.split(",")
        
        # If it has no commas, it is the trainer's name
        if len(arr) == 1:
            trainer = line.strip()
            continue

        # mon level item ability move 1, move 2, move 3, move 4
        # 0   1     2    3       4
        print(trainer + " (" + arr[0] + ") @ " + arr[2])
        print('Level: ' + arr[1])
        print("Serious Nature")
        print("Ability: " + arr[3])
        print("- " + arr[4])
        print("- " + arr[5])
        print("- " + arr[6])
        print("- " + arr[7])
        print("")


#trainer (species) @ held_item
#Level: level
#Serious Nature
#Ability: ability
#- move1
#- move2
#- move3
#- move4
