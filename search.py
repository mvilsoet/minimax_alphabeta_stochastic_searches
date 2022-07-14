import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moveList = []
    moveTree = {}
    if depth == 0:
      return evaluate(board), moveList, moveTree

    moves = []
    for move in generateMoves(side, board, flags):
      moves.append(move)
    if len(moves) == 0:
      return evaluate(board), moveList, moveTree
   
    bestmovelist = []
    bestval = -1*math.inf if not side else math.inf
    if side == False: #max
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        currval, currmovelist, currmovetree = minimax(newside, newboard, newflags, depth-1)
        moveTree[encode(*move)] = currmovetree
        
        if currval > bestval:
          bestval = currval
          bestmovelist = [move] + currmovelist

    else: #min
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        currval, currmovelist, currmovetree = minimax(newside, newboard, newflags, depth-1)
        moveTree[encode(*move)] = currmovetree

        if currval < bestval:
          bestval = currval
          bestmovelist = [move] + currmovelist

    moveList = bestmovelist
    return bestval, moveList, moveTree

def alphabeta(side, board, flags, depth, alpha=-1*math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moveTree = {}
    if depth == 0:
      return evaluate(board), [], moveTree

    moves = []
    for move in generateMoves(side, board, flags):
      moves.append(move)
    if len(moves) == 0:
      return evaluate(board), [], moveTree

    bestmovelist = []
    bestval = -1*math.inf if not side else math.inf
    if side == False: #max
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        currval, currmovelist, currmovetree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        moveTree[encode(*move)] = currmovetree

        if currval > bestval:
          bestval = currval
          bestmovelist = [move] + currmovelist

        alpha = max(alpha, bestval)
        if alpha >= beta:
          return alpha, bestmovelist, moveTree

    else: #min
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        currval, currmovelist, currmovetree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        moveTree[encode(*move)] = currmovetree

        if currval < bestval:
          bestval = currval
          bestmovelist = [move] + currmovelist

        beta = min(beta, bestval)
        if beta <= alpha:
          return beta, bestmovelist, moveTree

    return bestval, bestmovelist, moveTree
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moveList = []
    moveTree = {}
    if depth == 0: 
      return evaluate(board), moveList, moveTree
    if breadth == 0:
      return evaluate(board), moveList, moveTree

    moves = []
    for move in generateMoves(side, board, flags):
      moves.append(move)
    if len(moves) == 0:
      return evaluate(board), moveList, moveTree

    startvalues = []
    startmovelists = []
    for move in moves:
      initside, initboard, initflags = makeMove(side, board, move[0], move[1], flags, move[2])
      initmovetree = {}

      valuesum = 0
      for i in range(breadth):
        randval, dummy, randmovetree = breadthpath(initside, initboard, initflags, depth-1, chooser)
        valuesum += randval
        initmovetree.update(randmovetree)
      
      startvalues.append(valuesum/breadth)
      startmovelists.append([move])
      
      moveTree[encode(*move)] = initmovetree
    
    bestval = -1*math.inf if not side else math.inf
    if side == False: #max
      maxpath = None
      for i in range(len(startvalues)):
        if startvalues[i] > bestval:
          bestval = startvalues[i]
          maxpath = startmovelists[i]
      
      return bestval, maxpath, moveTree

    else: #min  
      minpath = None
      for i in range(len(startvalues)):
        if startvalues[i] < bestval:
          bestval = startvalues[i]
          minpath = startmovelists[i]
      
      return bestval, minpath, moveTree

def breadthpath(side, board, flags, depth, chooser):
  
  moveList = []
  moveTree = {}

  if depth == 0: 
    return evaluate(board), moveList, moveTree

  moves = []
  for move in generateMoves(side, board, flags):
    moves.append(move)
  if len(moves) == 0:
    return evaluate(board), moveList, moveTree
  
  randmove = chooser(moves)
  newside, newboard, newflags = makeMove(side, board, randmove[0], randmove[1], flags, randmove[2])
  pathval, pathlist, pathtree = breadthpath(newside, newboard, newflags, depth-1, chooser)

  moveTree[encode(*randmove)] = pathtree
  moveList = [randmove] + pathlist

  return pathval, moveList, moveTree
