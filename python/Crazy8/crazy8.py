import random

COLORS = ['r', 'y', 'g', 'b']
VALUES = ['0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', 's', 's','r', 'r', 'zd2', 'zd2', 'zd4', 'w']
WILDS = ['zd4', 'w']
TOTAL_ROUNDS = 10000
STARTING_HAND_SIZE = 20

class NaiveCard:
    def __init__(self):
        self._value = random.choice(VALUES)
        if self._value in WILDS:
            self._color = 'z'
        else:
            self._color = random.choice(COLORS)
    def __eq__(self, other):
        return self._color is other._color and self._value is other._value
    def matches(self, other):
        if self._value in WILDS:
            return True
        return self._value == other._value or self._color == other._color
    def __str__(self):
        return self._color + ":" + self._value
    def __gt__(self, other):
        if self._color == other._color:
            return self._value < other._value
        else:
            return self._color > other._color 
    def __lt__(self, other):
        if self._color == other._color:
            return self._value > other._value
        else:
            return self._color < other._color 

class NaiveHand:
    def __init__(self):
        self._cards = []
        self._wins = 0
        self.addCards(STARTING_HAND_SIZE)
    def __str__(self):
        return ', '.join([str(card) for card in self._cards])
    def size(self):
        return len(self._cards)
    def addCards(self, n):
        for i in range(n):
            self._cards.append(NaiveCard())
        self.reorder()
    def playNextCard(self, card, opponentHandSize):
        for c in self._cards:
            if c.matches(card):
                self._cards.remove(c)
                return c
        self.addCards(1)
        for c in self._cards:
            if c.matches(card):
                self._cards.remove(c)
                return c
    def reorder(self):
        self._cards.sort()

    def hasD4(self):
        for c in self._cards:
            if c._value is 'zd4':
                self._cards.remove(c)
                return True
        return False
    def hasD2(self):
        for c in self._cards:
            if c._value is 'zd2':
                self._cards.remove(c)
                return True
        return False
    def mostFrequentColor(self):
        counter = 0
        color = ''
        for c in self._cards:
            freq = 0
            for d in self._cards:
                if c._color is d._color:
                    freq+=1
            if(freq > counter):
                counter = freq
                color = c._color
        return color

class SmartHand:
    def __init__(self):
        self._cards = []
        self._wins = 0
        self.addCards(STARTING_HAND_SIZE)
    def __str__(self):
        return ', '.join([str(card) for card in self._cards])
    def size(self):
        return len(self._cards)
    def addCards(self, n):
        for i in range(n):
            self._cards.append(SmartCard())
        self.reorder()
    def playNextCard(self, card, opponentHandSize):
        # play aggressive if the other person has fewer cards
        if opponentHandSize < 2:
            self._cards.reverse()
            for c in self._cards:
                if c.matches(card):
                    self._cards.remove(c)
                    self._cards.reverse()
                    return c
            self._cards.reverse()
        else:
            for c in self._cards:
                if c.matches(card):
                    self._cards.remove(c)
                    return c
        self.addCards(1)
        for c in self._cards:
            if c.matches(card):
                self._cards.remove(c)
                return c
    def reorder(self):
        self._cards.sort()

    def hasD4(self):
        for c in self._cards:
            if c._value is 'zd4':
                self._cards.remove(c)
                return True
        return False
    def hasD2(self):
        for c in self._cards:
            if c._value is 'zd2':
                self._cards.remove(c)
                return True
        return False
    def mostFrequentColor(self):
        counter = 0
        color = ''
        for c in self._cards:
            freq = 0
            for d in self._cards:
                if c._color is d._color:
                    freq+=1
            if(freq > counter):
                counter = freq
                color = c._color
        return color

class SmartCard:
    def __init__(self):
        self._value = random.choice(VALUES)
        if self._value in WILDS:
            self._color = 'z'
        else:
            self._color = random.choice(COLORS)
    def __eq__(self, other):
        return self._color is other._color and self._value is other._value
    def matches(self, other):
        if self._value in WILDS:
            return True
        return self._value == other._value or self._color == other._color
    def __str__(self):
        return self._color + ":" + self._value
    def __gt__(self, other):
        if self._color == other._color:
            return self._value > other._value
        else:
            return self._color > other._color 
    def __lt__(self, other):
        if self._color == other._color:
            return self._value < other._value
        else:
            return self._color < other._color 

def resetGame():
    pass

def runGame():
    p1 = NaiveHand()
    p2 = SmartHand()
    p1Turn = True

    p1Wins = 0
    p2Wins = 0
    for roundNumber in range(TOTAL_ROUNDS):
        # reset the current game
        gameOver = False
        currentCard = NaiveCard()
        while(currentCard._value in WILDS):
            currentCard = NaiveCard()

        # game cycle
        while(not gameOver):
            if(p1Turn):
                card = p1.playNextCard(currentCard, p2.size())
                if card is None:
                    continue
                currentCard = card
                #print("p1 plays: " + str(card))
                #print("p1 hand: " + str(p1))
                if p1.size() is 0:
                    p1Wins += 1
                    gameOver = True
                    p1Turn = not p1Turn
                    break

                if card._value in WILDS:
                    # choose a color based on your most frequent color
                    card._color = p1.mostFrequentColor()
                if card._value is 'zd4':
                    while(True):
                        if not p2.hasD4():
                            p2.addCards(4)
                            p1Turn = not p1Turn
                            break
                        elif not p1.hasD4():
                            p1.addCards(4)
                            break

                if card._value is 'zd2':
                    while(True):
                        if not p2.hasD2():
                            p2.addCards(2)
                            p1Turn = not p1Turn
                            break
                        elif not p1.hasD2():
                            p1.addCards(2)
                            break

                if card._value in ['s','r']:
                    p1Turn = not p1Turn
            else:
                #p2 turn
                card = p2.playNextCard(currentCard, p1.size())
                if card is None:
                    continue
                currentCard = card
                #print("p2 plays: " + str(card))
                #print("p2 hand: " + str(p2))
                if p2.size() is 0:
                    p2Wins += 1
                    gameOver = True
                    p1Turn = not p1Turn
                    break

                if card._value in WILDS:
                    # choose a color based on your most frequent color
                    card._color = p2.mostFrequentColor()
                if card._value is 'zd4':
                    while(True):
                        if not p1.hasD4():
                            p1.addCards(4)
                            p1Turn = not p1Turn
                            break
                        elif not p2.hasD4():
                            p2.addCards(4)
                            break

                if card._value is 'zd2':
                    while(True):
                        if not p1.hasD2():
                            p1.addCards(2)
                            p1Turn = not p1Turn
                            break
                        elif not p2.hasD2():
                            p2.addCards(2)
                            break

                if card._value in ['s','r']:
                    p1Turn = not p1Turn
            p1Turn = not p1Turn
    print("P1 wins: " + str(p1Wins))
    print("P2 wins: " + str(p2Wins))

def main():
    runGame()

if __name__ == "__main__":
    main()
