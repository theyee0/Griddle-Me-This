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

    board_tensor = chess_to_tensor(board)

    # Iterate over all squares to identify valid moves
    with torch.no_grad():
        probability_from = {piece : models_from[piece](board_tensor) for piece in chess_pieces}

        for from_square in chess.SQUARES:
            piece = board.piece_at(from_square)

            # Move onto the next square if the current square is empty or opposing color
            if piece is None or piece.color == chess.BLACK:
                continue

            piece = piece.piece_type

            # Compute probability that the ideal move is to move the current piece
            from_probability = probability_from[piece].data[0].data[from_square].item()

            moves = filter(lambda x: x.from_square == from_square, board.legal_moves)

            # Iterate over valid moves
            for move in moves:
                to_square = move.to_square

                # Compute probability that a given move will end up here
                to_probability = models_to[piece](board_tensor).data[0].data[to_square].item()

                combined_probability = from_probability * to_probability

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
            move = chess.Move.from_uci(input("Enter a move: "))
        except chess.InvalidMoveError:
            # Moves is not in valid UCI format
            print("Please format your move properly.")
            move = None

        # Move is illegal
        if not board.is_legal(move):
            print("Please enter a valid move.")
            move = None

    return move


def game_loop(board, models_from, models_to):
    """Create infinite game loop where the user enters a move and the network responds"""
    print(board)

    while True:
        player_move = read_move(board)
        print(f"Your move was {player_move}")
        board.push(player_move)
        if board.is_game_over():
            printf("The player won a round!")
            print(board.outcome().result())
            board.reset()
        print(board)
        print()

        board.apply_mirror()
        computer_move = predict_move(board, models_from, models_to)
        mirrored_move = chess.Move(chess.square_mirror(computer_move.from_square),
                                   chess.square_mirror(computer_move.to_square))
        print(f"The computer played {mirrored_move}")
        board.push(computer_move)
        board.apply_mirror()
        if board.is_game_over():
            printf("The computer won the round!")
            print(board.outcome().result())
            board.reset()
        print(board)
        print()


def load_models():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    models_from = {}
    models_to = {}

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    for i, piece in enumerate(chess_pieces):
        models_from[piece] = StackedConvolve().to(device)
        models_from[piece].load_state_dict(torch.load(f"model_from_{i}.pt"))

        models_to[piece] = StackedConvolve().to(device)
        models_to[piece].load_state_dict(torch.load(f"model_to_{i}.pt"))

    return models_from, models_to


if __name__ == "__main__":
    models_from, models_to = load_models()
    game_loop(chess.Board(), models_from, models_to)

