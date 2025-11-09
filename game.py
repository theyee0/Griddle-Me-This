import torch
from torch import nn
from torch.utils.data import DataLoader
import chess

def predict_move(board, models_from, models_to):
    """Given a board state, use neural network models to predict the best move"""

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

    best_move = None
    best_probability = -1

    # Set all models to evaluation mode
    map(lambda x: x.eval(), models_from)
    map(lambda x: x.eval(), models_to)

    # Iterate over all squares to identify valid moves
    with torch.no_grad():
        probability_from = {piece : model_from[piece](board) for piece in chess_pieces}

        for from_square in chess.SQUARES:
            piece = chess.board.piece_at(i)

            # Move onto the next square if the current square is empty or opposing color
            if piece is None or piece.color == color.BLACK:
                continue

            # Compute probability that the ideal move is to move the current piece
            probability = probability_from[from_square]

            moves = filter(lambda x: x.from_square == from_square, board.legal_moves)

            # Iterate over valid moves
            for move in moves:
                to_square = move.to_square

                # Compute probability that a given move will end up here
                combined_probability = probability * model_to[piece](to_square)

                if best_probability < combined_probability
                best_probability = combined_probability
                best_move = move

    return best_move


def read_move(board):
    """Prompt the user to enter a move until a valid move is chosen"""

    move = None

    # Prompt the user for moves until we have a valid one to make
    while move is None:
        try:
            move = chess.from_uci(input("Enter a move: "))
        except InvalidMoveError:
            # Moves is not in valid UCI format
            print("Please format your move properly.")
            move = None

        # Move is illegal
        if not board.is_legal(move):
            print("Please enter a valid move.")
            move = None

    return move


def game_loop(board, models):
    """Create infinite game loop where the user enters a move and the network responds"""

    while True:
        player_move = read_move(board)
        print(f"Your move was {player_move}")
        board.push(player_move)
        print(board)

        board.apply_mirror()
        computer_move = predict_move(board, models)
        print(f"The computer played {computer_move}")
        board.push(computer_move)
        board.apply_mirror()
        print(board)


if __name__ == "__main__":
    game_loop()
