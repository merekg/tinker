import json
import sys

if __name__ == "__main__":
    jsonPath = sys.argv[1]
    jsonFile = open(jsonPath)
    data = json.load(jsonFile)
    bodypressfile = open(sys.argv[2])
    bodypresslist = []
    for line in bodypressfile:
        bodypresslist.append(line.rstrip())
    maxdef = 0
    maxdefmon =''
    for i in data:
        name = i['name']['english']
        defense = i['base']['Defense']
        spdef = i['base']['Sp. Defense']
        t = i['type']
        if spdef + defense > maxdef:
            maxdefmon = name
            maxdef = defense + spdef
    print(maxdefmon)
    print(maxdef)
