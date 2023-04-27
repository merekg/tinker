import sys
import operator
import os

class Mon:
    def __init__(self, initString):
        self.string = initString.lower().split(',')
        self.stats = []
        for i in range(6):
            self.stats.append(int(self.string[i+1].split("=")[1]))

    def __str__(self):
        return self.string[0]
    def getstats(self):
        return self.stats
    def getBst(self):
        return sum(self.stats)
    def getHeadStats(self):
        return .667*self.stats[0] + .667*self.stats[3] + .667*self.stats[4] + .333*self.stats[1] + .333*self.stats[2] + .333*self.stats[5]
    def getBodyStats(self):
        return .667*self.stats[1] + .667*self.stats[2] + .667*self.stats[5] + .333*self.stats[0] + .333*self.stats[3] + .333*self.stats[4]
    # monName = (level, move1), (level, move2)
    def setMoves(self,moveString):
        moveAndLevelArray = moveString.split('=')[1].split(';')


def main():
    assert len(sys.argv) is 3
    statsFile = open(sys.argv[1], 'r')

    bestHeadTotal = 0
    bestHead = ""
    bestBodyTotal = 0
    bestBody = ""
    for line in statsFile:
        mon = Mon(line)
        if mon.getHeadStats() > bestHeadTotal:
            bestHeadTotal = mon.getHeadStats()
            bestHead = str(mon)
        if mon.getBodyStats() > bestBodyTotal:
            bestBodyTotal = mon.getBodyStats()
            bestBody = str(mon)
    print(bestHead)
    print(bestHeadTotal)
    print(bestBody)
    print(bestBodyTotal)

if __name__ == "__main__":
    main()
