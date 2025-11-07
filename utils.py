import torch
from torch import nn
from torch.utils.data import DataLoader
import chess

def chess_to_tensor(board):
    types = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING)
    colors = (chess.WHITE, chess.BLACK)

    pieces = map(chess.board.piece_at, chess.SQUARES)

    piece_selector = lambda x: lambda y: 1 if x == y else 0
    piece_tensor = lambda x, y: torch.Tensor(map(piece_selector(chess.Piece(x, y)), pieces))

    subtensors = [torch.chunk(piece_tensor(x, y), 8) for x in types for y in colors]

    return torch.Tensor(subtensors)


def tensor_to_chess(tensor):
    board = chess.Board()
    types = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING)
    colors = (chess.WHITE, chess.BLACK)

    board_type = [(x, y) for x in types for y in colors]

    board.clear()

    for i, subtensor in enumerate(tensor):
        piece, color = board_type[i]

        for j, row in enumerate(subtensor):
            for k, col in enumerate(row):
                square = chess.square(k + 1, j + 1)

                if col == 1:
                    board.set_piece_at(square, chess.Piece(piece, color))

    return board
                    

def position_to_tensor(position):
    """Returns a one-hot encoded flat tensor representation of a position"""
    pieces = map(lambda x: position == x, chess.SQUARES)
    return torch.Tensor(pieces)


def tensor_to_position(tensor):
    """Given a flat tensor with 64 locations representing a one-hot board, return the square"""

    assert len(tensor) == 64, "Expected valid 8x8 chess board as a flat tensor"
    matches = torch.nonzero(tensor)
    assert len(matches) == 1, "Expected one-hot encoding for position tensor"

    return chess.SQUARES[matches[0]]
