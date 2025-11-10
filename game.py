import torch
from model import *
from utils import *
import chess

def predict_move(board, models_from, models_to):
    """Given a board state, use neural network models to predict the best move"""

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    best_move = None
    best_probability = -1

    # Set all models to evaluation mode
    for _, model in models_from.items():
        model.eval()
    for _, model in models_to.items():
        model.eval()

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

                if best_probability < combined_probability:
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


def load_models():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    models_from = {}
    models_to = {}

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    for i, piece in enumerate(chess_pieces):
        model_from[piece] = StackedConvolve.to(device)
        model_from.load_state_dict(torch.load("model_from_{i}.pt"))

        model_to[piece] = StackedConvolve.to(device)
        model_to.load_state_dict(torch.load("model_to_{i}.pt"))

    return models_from, models_to


if __name__ == "__main__":
    models_from, models_to = load_models()
    game_loop(chess.Board(), models_from, models_to)

