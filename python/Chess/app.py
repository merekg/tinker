import tkinter as tk
from dataclasses import dataclass
from enum import Enum

class Color(Enum):
    NONE = 0
    WHITE = 1
    BLACK = 2

class Sprite(Enum):
    WHITE_KING = "\u2654"
    WHITE_QUEEN = "\u2655"
    WHITE_ROOK = "\u2656"
    WHITE_BISHOP = "\u2657"
    WHITE_KNIGHT = "\u2658"
    WHITE_PAWN = "\u2659"
    BLACK_KING = "\u265A"
    BLACK_QUEEN ="\u265B"
    BLACK_ROOK ="\u265C"
    BLACK_BISHOP ="\u265D"
    BLACK_KNIGHT ="\u265E"
    BLACK_PAWN ="\u265F"

class Pose:
    def __new__(cls, row, col):
        if row in range(8) and col in range(8):
            instance = super(Pose, cls).__new__(cls)
            instance.row, instance.column = row, col
            return instance
    def __str__(self):
        return str(self.row) + ", " + str(self.column)
    def __eq__(self, other):
        if isinstance(other, Pose):
            return self.row == other.row and self.column == other.column
        return False
    # hash is unique for each row /  column combination
    def __hash__(self): 
        return 2**self.row * 3**self.column


class Piece:

    def __init__(self, color, initialPose):
        self.pose = initialPose
        self.color = color
        self.hasMoved = False
        self.isCaptured = False
    def __str__(self):
        return self.getSprite() + " at " + str(self.pose)
    def getPose(self):
        return self.pose
    def setPose(self, pose, board):
        if(not self.isValidMove(pose, board)):
            return False
        self.hasMoved = True
        self.pose = pose
        return True
    def isValidMove(self, move, board):
        return move in self.getAllValidMoves(board)
    def getAllValidMoves(self, board):
        return set()
    def getSprite(self):
        pass
    def getColor(self):
        return self.color
    def isWhite(self):
        return self.color == Color.WHITE
    def getHasMoved(self):
        return self.hasMoved
    def capture(self):
        self.pose = None
        self.isCaptured = True
    def getCaptured(self):
        return self.isCaptured

class King(Piece):
    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)

    def getSprite(self):
        return Sprite.WHITE_KING.value if self.color == Color.WHITE else Sprite.BLACK_KING.value

    def getAllValidMoves(self, board):
        validMoves = set()
        if not board.isSameColor(Pose(self.pose.row + 1, self.pose.column), self.color):
            validMoves.add(Pose(self.pose.row + 1, self.pose.column))
        if not board.isSameColor(Pose(self.pose.row + 1, self.pose.column - 1), self.color):
            validMoves.add(Pose(self.pose.row + 1, self.pose.column - 1))
        if not board.isSameColor(Pose(self.pose.row + 1, self.pose.column + 1), self.color):
            validMoves.add(Pose(self.pose.row + 1, self.pose.column + 1))
        if not board.isSameColor(Pose(self.pose.row - 1, self.pose.column), self.color):
            validMoves.add(Pose(self.pose.row - 1, self.pose.column))
        if not board.isSameColor(Pose(self.pose.row - 1, self.pose.column - 1), self.color):
            validMoves.add(Pose(self.pose.row - 1, self.pose.column - 1))
        if not board.isSameColor(Pose(self.pose.row - 1, self.pose.column + 1), self.color):
            validMoves.add(Pose(self.pose.row - 1, self.pose.column + 1))
        if not board.isSameColor(Pose(self.pose.row, self.pose.column), self.color):
            validMoves.add(Pose(self.pose.row, self.pose.column))
        if not board.isSameColor(Pose(self.pose.row, self.pose.column - 1), self.color):
            validMoves.add(Pose(self.pose.row, self.pose.column - 1))
        if not board.isSameColor(Pose(self.pose.row, self.pose.column + 1), self.color):
            validMoves.add(Pose(self.pose.row, self.pose.column + 1))

    def getProtectedSquares(self, board):
        return self.getAllValidMoves(board)

    def castle(self, kingSide):
        if kingSide:
            self.pose = Pose(0,6)
        else:
            self.pose = Pose(0,2)

class Queen(Piece):
    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)

    def getSprite(self):
        return Sprite.WHITE_QUEEN.value if self.color == Color.WHITE else Sprite.BLACK_QUEEN.value
    
    def getAllValidMoves(self, board):
        validMoves = set()
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row, self.pose.column + i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column + i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column + i), self.color):
                break
        return validMoves

    def getProtectedSquares(self, board):
        return self.getAllValidMoves(board)

class Rook(Piece):
    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)

    def getSprite(self):
        return Sprite.WHITE_ROOK.value if self.color == Color.WHITE else Sprite.BLACK_ROOK.value
    
    def getAllValidMoves(self, board):
        validMoves = set()
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row, self.pose.column + i), self.color):
                break
        return validMoves

    def getProtectedSquares(self, board):
        return self.getAllValidMoves(board)

class Bishop(Piece):

    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)

    def getSprite(self):
        return Sprite.WHITE_BISHOP.value if self.color == Color.WHITE else Sprite.BLACK_BISHOP.value
    
    def getAllValidMoves(self, board):
        validMoves = set()
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row - i, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row - i, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row - i, self.pose.column + i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column - i), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column - i))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column - i), self.color):
                break
        for i in range(1,8):
            if board.isSameColor(Pose(self.pose.row + i, self.pose.column + i), self.color):
                break
            validMoves.add(Pose(self.pose.row + i, self.pose.column + i))
            if board.isOpposingColor(Pose(self.pose.row + i, self.pose.column + i), self.color):
                break
        return validMoves

    def getProtectedSquares(self, board):
        return self.getAllValidMoves( board)

class Knight(Piece):

    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)

    def getSprite(self):
        return Sprite.WHITE_KNIGHT.value if self.color == Color.WHITE else Sprite.BLACK_KNIGHT.value
    
    def getAllValidMoves(self, board):
        validMoves = set()
        if not board.isSameColor(Pose(self.pose.row - 2, self.pose.column - 1), self.color):
            validMoves.add(Pose(self.pose.row - 2, self.pose.column - 1))
        if not board.isSameColor(Pose(self.pose.row + 2, self.pose.column + 1), self.color):
            validMoves.add(Pose(self.pose.row + 2, self.pose.column + 1))
        if not board.isSameColor(Pose(self.pose.row - 2, self.pose.column + 1), self.color):
            validMoves.add(Pose(self.pose.row - 2, self.pose.column + 1))
        if not board.isSameColor(Pose(self.pose.row + 2, self.pose.column - 1), self.color):
            validMoves.add(Pose(self.pose.row + 2, self.pose.column - 1))
        if not board.isSameColor(Pose(self.pose.row - 1, self.pose.column - 2), self.color):
            validMoves.add(Pose(self.pose.row - 1, self.pose.column - 2))
        if not board.isSameColor(Pose(self.pose.row + 1, self.pose.column + 2), self.color):
            validMoves.add(Pose(self.pose.row + 1, self.pose.column + 2))
        if not board.isSameColor(Pose(self.pose.row - 1, self.pose.column + 2), self.color):
            validMoves.add(Pose(self.pose.row - 1, self.pose.column + 2))
        if not board.isSameColor(Pose(self.pose.row + 1, self.pose.column - 2), self.color):
            validMoves.add(Pose(self.pose.row + 1, self.pose.column - 2))
        return validMoves

    def getProtectedSquares(self, board):
        return self.getAllValidMoves(board)

class Pawn(Piece):
    def __init__(self, color, initialPose):
        super().__init__(color, initialPose)
        self.doubleStep = False
        self.enpassantCaptures = set()

    def getSprite(self):
        return Sprite.WHITE_PAWN.value if self.color == Color.WHITE else Sprite.BLACK_PAWN.value

    def setPose(self, pose, board):
        if(not self.isValidMove(pose, board)):
            return False
        if self.isDoubleMove(pose):
            self.doubleStep = True
        self.hasMoved = True
        if pose in self.enpassantCaptures:
            if self.color == Color.WHITE:
                board.capture(Pose(pose.row -1, pose.column))
            else:
                board.capture(Pose(pose.row +1, pose.column))
        self.enpassantCaptures = set()
        self.pose = pose
        return True

    def isDoubleMove(self, pose):
        return abs(self.pose.row - pose.row) > 1

    def getDoubleStep(self):
        return self.doubleStep

    def resetDoubleStep(self):
        self.doubleStep = False;
    
    def getAllValidMoves(self, board):
        validMoves = set()
        if not self.hasMoved:
            if board.isEmpty(Pose(self.pose.row + 1, self.pose.column)) and board.isEmpty(Pose(self.pose.row +2, self.pose.column)):
                validMoves.add(Pose(self.pose.row + (2 if self.color == Color.WHITE else -2), self.pose.column))

        if board.isEmpty(Pose(self.pose.row + 1, self.pose.column)):
            validMoves.add(Pose(self.pose.row + (1 if self.color == Color.WHITE else -1), self.pose.column))

        # captures
        if self.color == Color.WHITE:
            if board.isOpposingColor(Pose(self.pose.row + 1, self.pose.column -1), self.color):
                validMoves.add(Pose(self.pose.row + 1, self.pose.column + 1))
            if board.isOpposingColor(Pose(self.pose.row - 1, self.pose.column +1), self.color):
                validMoves.add(Pose(self.pose.row - 1, self.pose.column + 1))
            if self.pose.row == 4:
                # check the two en pessant squares
                p1 = board.getPieceAt(Pose(4, self.pose.column-1))
                p2 = board.getPieceAt(Pose(4, self.pose.column+1))
                if isinstance(p1, Pawn) and p1.doublesStep:
                    validMoves.add(Pose(self.pose.row + 1, self.pose.column-1))
                    self.enpassantCaptures.add(Pose(self.pose.row + 1, self.pose.column-1))
                if isinstance(p2, Pawn) and p2.doublesStep:
                    validMoves.add(Pose(self.pose.row + 1, self.pose.column+1))
                    self.enpassantCaptures.add(Pose(self.pose.row + 1, self.pose.column+1))
        if self.color == Color.BLACK:
            if board.isOpposingColor(Pose(self.pose.row + 1, self.pose.column -1), self.color):
                validMoves.add(Pose(self.pose.row + 1, self.pose.column + 1))
            if board.isOpposingColor(Pose(self.pose.row - 1, self.pose.column +1), self.color):
                validMoves.add(Pose(self.pose.row - 1, self.pose.column + 1))
            if self.pose.row == 3:
                # check the two en pessant squares
                p1 = board.getPieceAt(Pose(3, self.pose.column-1))
                p2 = board.getPieceAt(Pose(3, self.pose.column+1))
                if isinstance(p1, Pawn) and p1.doublesStep:
                    validMoves.add(Pose(self.pose.row - 1, self.pose.column-1))
                    self.enpassantCaptures.add(Pose(self.pose.row - 1, self.pose.column-1))
                if isinstance(p2, Pawn) and p2.doublesStep:
                    validMoves.add(Pose(self.pose.row - 1, self.pose.column+1))
                    self.enpassantCaptures.add(Pose(self.pose.row - 1, self.pose.column+1))
        return validMoves

    def getProtectedSquares(self, board):
        validMoves = set()
        if self.color == Color.WHITE:
            if board.isOpposingColor(Pose(self.pose.row + 1, self.pose.column -1), self.color):
                validMoves.add(Pose(self.pose.row + 1, self.pose.column - 1))
            if board.isOpposingColor(Pose(self.pose.row + 1, self.pose.column +1), self.color):
                validMoves.add(Pose(self.pose.row + 1, self.pose.column + 1))
        if self.color == Color.BLACK:
            if board.isOpposingColor(Pose(self.pose.row - 1, self.pose.column -1), self.color):
                validMoves.add(Pose(self.pose.row - 1, self.pose.column - 1))
            if board.isOpposingColor(Pose(self.pose.row - 1, self.pose.column +1), self.color):
                validMoves.add(Pose(self.pose.row - 1, self.pose.column + 1))
        return validMoves

class Board:
    def __init__(self):
        self.pieces = set()
        # the kings are named pieces 
        self.whiteKing = King(Color.WHITE, Pose(0,4))
        self.blackKing = King(Color.BLACK, Pose(7,4))
        # the rest are just pieces
        self.pieces.add(Pawn(Color.WHITE, Pose(1,0)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,1)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,2)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,3)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,4)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,5)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,6)))
        self.pieces.add(Pawn(Color.WHITE, Pose(1,7)))
        self.pieces.add(Rook(Color.WHITE,   Pose(0,7)))
        self.pieces.add(Rook(Color.WHITE,   Pose(0,0)))
        self.pieces.add(Knight(Color.WHITE, Pose(0,1)))
        self.pieces.add(Knight(Color.WHITE, Pose(0,6)))
        self.pieces.add(Bishop(Color.WHITE, Pose(0,5)))
        self.pieces.add(Bishop(Color.WHITE, Pose(0,2)))
        self.pieces.add(Queen(Color.WHITE,  Pose(0,3)))
        self.pieces.add(self.whiteKing)
        self.pieces.add(Pawn(Color.BLACK, Pose(6,0)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,1)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,2)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,3)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,4)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,5)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,6)))
        self.pieces.add(Pawn(Color.BLACK, Pose(6,7)))
        self.pieces.add(Rook(Color.BLACK,   Pose(7,7)))
        self.pieces.add(Rook(Color.BLACK,   Pose(7,0)))
        self.pieces.add(Knight(Color.BLACK, Pose(7,1)))
        self.pieces.add(Knight(Color.BLACK, Pose(7,6)))
        self.pieces.add(Bishop(Color.BLACK, Pose(7,5)))
        self.pieces.add(Bishop(Color.BLACK, Pose(7,2)))
        self.pieces.add(Queen(Color.BLACK,  Pose(7,3)))
        self.pieces.add(self.blackKing)

    def __str__(self):
        returnString = ''
        for p in self.pieces:
            if not p is None:
                returnString += str(p) + "\n"
        return returnString

    def getColorAt(self, pose):
        for piece in self.pieces:
            if piece.pose == pose:
                return piece.getColor()
        return None

    def getPieceAt(self, pose): 
        for piece in self.pieces:
            if piece.pose == pose:
                return piece
        return None

    def isBlack(self, pose):
        return self.getColorAt(pose) == Color.BLACK

    def isWhite(self, pose):
        return self.getColorAt(pose) == Color.WHITE

    def isEmpty(self, pose):
        return self.getColorAt(pose) == None

    def isSameColor(self, pose, color):
        return self.getColorAt(pose) == color

    def isOpposingColor(self, pose, color):
        return self.getColorAt(pose) != None and self.getColorAt(pose) != color
    
    def capture(self, pose):
        getPieceAt(pose).capture()

    def movePiece(currentPose, nextPose):
        getPieceAt(currentPose).setPose(nextPose)

    def getThreatMap(self, color):
        coveredSquares = set()
        for piece in self.pieces:
            if piece.color is color and not isinstance(piece, King):
                piecesquares = piece.getProtectedSquares(self)
                coveredSquares.update(piece.getProtectedSquares(self))
        return coveredSquares

    def castle(self, color, isKingSide):
        # check if the king has moved
        # check if the rook has moved
        # check if there are pieces in between the rook and the king
        # check if the king moves through check
        if color is Color.WHITE:
            if self.whiteKing.getHasMoved():
                return False
            if isKingSide:
                rook = self.getPieceAt(Pose(0,7))
                f1 = self.getPieceAt(Pose(0,5))
                g1 = self.getPieceAt(Pose(0,6))
                threatmap = self.getThreatMap(Color.BLACK)
                passesThroughCheck = Pose(0,4) in threatmap or Pose(0,5) in threatmap or Pose(0,6) in threatmap
                if rook is None or rook.hasMoved() or f1 is not None or g1 is not None or passesThroughCheck:
                    return False
                rook.setPose(Pose(0,5))
            else: # queenside
                rook = self.getPieceAt(Pose(0,0))
                b1 = self.getPieceAt(Pose(0,1))
                c1 = self.getPieceAt(Pose(0,2))
                d1 = self.getPieceAt(Pose(0,3))
                threatmap = self.getThreatMap(Color.BLACK)
                passesThroughCheck = Pose(0,2) in threatmap or Pose(0,3) in threatmap or Pose(0,4) in threatmap
                if rook is None or rook.hasMoved() or b1 is not None or c1 is not None or d1 is not None or passesThroughCheck:
                    return False
                rook.setPose(Pose(0,3))
            self.whiteKing.castle()
        else:
            if self.blackKing.getHasMoved():
                return False
            if isKingSide:
                rook = self.getPieceAt(Pose(7,7))
                f8 = self.getPieceAt(Pose(7,5))
                g8 = self.getPieceAt(Pose(7,6))
                threatmap = self.getThreatMap(Color.WHITE)
                passesThroughCheck = Pose(7,4) in threatmap or Pose(7,5) in threatmap or Pose(7,6) in threatmap
                if rook is None or rook.hasMoved() or f8 is not None or g8 is not None or passesThroughCheck:
                    return False
                rook.setPose(Pose(7,5))
            else: # queenside
                rook = self.getPieceAt(Pose(7,0))
                b1 = self.getPieceAt(Pose(7,1))
                c1 = self.getPieceAt(Pose(7,2))
                d1 = self.getPieceAt(Pose(7,3))
                threatmap = self.getThreatMap(Color.WHITE)
                passesThroughCheck = Pose(7,2) in threatmap or Pose(7,3) in threatmap or Pose(7,4) in threatmap
                if rook is None or rook.hasMoved() or b1 is not None or c1 is not None or d1 is not None or passesThroughCheck:
                    return False
                rook.setPose(Pose(7,3))
            self.blackKing.castle()

    def movePiece(self, currentPose, nextPose):
        for piece in self.pieces:
            if piece.pose == currentPose:
                piece.setPose(nextPose, self)
                return True
        return False

        for move in validMoves:
            if move in board.getThreatMap(Color.BLACK if self.color == Color.WHITE else Color.WHITE):
                validMoves = validMoves - set([move])
        return validMoves

    def resultsInCheck(self, piece, move):
        previousPose = piece.getPose()
        if not piece.setPose(move):
            return false
        rv = self.isCheck(piece.color)
        piece.setPose(previousPose)
        return rv

    def checkForValidMoves(self, color):
        validMoves = set()
        for piece in self.pieces:
            if piece.color is color:
                for move in piece.getAllValidMoves():
                    if not self.resultsInCheck(piece, move):
                        validMoves.add(tuple(piece, move))
        return validMoves
                    
    def isStalemate(self, color):
        return len(self.checkForValidMoves(color)) < 1
    
    def isCheck(self, color):
        if color == Color.WHITE:
            return self.whiteKing.getPose() in self.getThreatMap(Color.BLACK)
        else:
            return self.blackKing.getPose() in self.getThreatMap(Color.WHITE)

    def isCheckmate(self, color):
        return self.isCheck(color) and self.isStalemate(color)

    def insufficientMaterial(self):
        # Cases with insufficientMaterial:
        # no queens, no rooks, no pawns, fewer than 2 minor pieces which are not both knights and not both the same color bishop
        whiteMinorPieces = set()
        blackMinorPieces = set() 
        for piece in self.pieces:
            if isinstance(piece, (Queen, Rook, Pawn)):
                return False
            if isinstance(piece, (Knight, Bishop)):
                if piece.isWhite():
                    whiteMinorPieces.add(piece)
                else:
                    blackMinorPieces.add(piece)
        if len(whiteMinorPieces)<2 and len(blackMinorPieces)<2:
            return True
        return False

def char_range(c1, c2):
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

def draw():
    print("Draw")

def turnLoop(board):
    color = Color.WHITE
    movesSinceLastPawnMove = 0
    while(True):
        # Check for Checkmate
        if board.isCheckmate(color):
            print(color, "wins!")
            return
        # Check for stalemate
        if board.isStalemate(color):
            draw()
            return
        # check for 50 move draw
        if movesSinceLastPawnMove > 50:
            draw()
            return
        # check for insufficent material draw
        if board.insufficientMaterial():
            draw()
            return
        # check for 3 move repitition draw
        # prompt for move command
            # Regular move
            # Resignation
        # increment pawn move counter
        # reset en-pessant booleans
        color = Color.BLACK if color is Color.WHITE else Color.WHITE

if __name__ == "__main__":
    window = tk.Tk()

    # set up the buttons
    buttonFrame = tk.Frame(master=window)
    newGameButton = tk.Button(buttonFrame,text ="New game")
    newGameButton.pack(side=tk.LEFT)
    resignButton = tk.Button(buttonFrame,text ="Resign")
    resignButton.pack(side=tk.LEFT)
    buttonFrame.pack(side=tk.BOTTOM)

    # Set up the frames and the grid
    boardFrame = tk.Frame(master=window)
    for row in range(0,8):
        rowFrame = tk.Frame(master=boardFrame)
        for column in range(0,8):
            # alternate each color in checker pattern
            frame = tk.Frame(master=rowFrame, width=100, height=100, bg = ('black' if (row+column)%2 else 'white'))
            frame.pack(side=tk.LEFT)
        rowFrame.pack()
    boardFrame.pack(side=tk.TOP)

    board = Board()
    print(board)
    #tm = board.getThreatMap(Color.WHITE)
    #for p in tm:
        #print(p)

    board.movePiece(Pose(1,4), Pose(3,4))
    #print(board)
    tm = board.getThreatMap(Color.WHITE)
    for p in tm:
        print(p)

    #window.mainloop()
