from model import StackedConvolve
from utils import position_to_tensor, chess_to_tensor
import torch
from torch import nn
from torch.utils.data import DataLoader
import chess
import chess.pgn

class ChessDataset(torch.utils.data.Dataset):
    """Torch dataset containing (piece, board state, from square, to square)"""

    def __init__(self, move_pairs):
        self.board_state: list[torch.Tensor] = []
        self.squares_from: list[torch.Tensor] = []
        self.squares_to: list[torch.Tensor] = []
        self.pieces: list[chess.Piece] = []

        # Extract data from board state and moves
        for board, move in move_pairs:
            self.board_state.append(chess_to_tensor(board))
            self.squares_from.append(position_to_tensor(move.from_square))
            self.squares_to.append(position_to_tensor(move.to_square))
            self.pieces.append(board.piece_at(move.from_square).piece_type)

    def __len__(self):
        return len(self.board_state)

    def __getitem__(self, idx):
        return self.board_state[idx], self.squares_from[idx], self.squares_to[idx], self.pieces[idx]


def load_data(pgn_file):
    """Loads a PGN file and stores each move in (board state, move) pairs"""

    board = chess.Board()

    pgn = open(pgn_file)

    game = chess.pgn.read_game(pgn)
    game_tuples = []

    # Iterate as long as a game can be found in the pgn file
    while game is not None:
        # Start each game with a standard board layout
        board.reset()

        # Simulate each move and add it to the list
        for move in game.mainline_moves():
            if board.turn == chess.WHITE:
                game_tuples.append((board.copy(), move))
            else:
                mirrored_move = chess.Move(
                    chess.square_mirror(move.from_square),
                    chess.square_mirror(move.to_square))
                game_tuples.append((board.mirror(), mirrored_move))

            board.push(move)

        # Attempt to read another game
        game = chess.pgn.read_game(pgn)

    return game_tuples


def train(dataloader, models_from, models_to, loss_fn, optimizers_from, optimizers_to, device):
    """Trains two lists of models based on a dataset using pre-specified optimizers"""

    # Set all models to training mode
    for _, model in models_from.items():
        model.train()
    for _, model in models_to.items():
        model.train()

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    for batch, (X, y, z, piece) in enumerate(dataloader):
        X, y, z = X.to(device), y.to(device), z.to(device)

        # TODO: Fix goofy trick to avoid using tensor in dataloader
        piece = piece[0].item()

        # Train model corresponding to the currently moved piece that identifies "from" square
        pred_from = models_from[piece](X)
        loss_from = loss_fn(pred_from, y)

        loss_from.backward()
        optimizers_from[piece].step()
        optimizers_from[piece].zero_grad()

        # Train model corresponding to the currently moved piece that identifies "to" square
        pred_to = models_to[piece](X)
        loss_to = loss_fn(pred_to, z)

        loss_to.backward()
        optimizers_to[piece].step()
        optimizers_to[piece].zero_grad()


def train_on_games(pgn_file, models_from, models_to, epochs, device):
    """Loads a dataset from a pgn file and trains the lists of models for some epochs"""

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    game_pairs = load_data(pgn_file)
    dataset = ChessDataset(game_pairs)
    loader = DataLoader(dataset)

    # Define loss function and create list of optimizers for each model in lists
    loss_fn = nn.CrossEntropyLoss()

    optimizers_from = {}
    optimizers_to = {}

    for piece in chess_pieces:
        optimizers_from[piece] = torch.optim.SGD(models_from[piece].parameters(), lr=1e-3)
        optimizers_to[piece] = torch.optim.SGD(models_to[piece].parameters(), lr=1e-3);

    # Iteratively train over epochs
    for epoch in range(epochs):
        print(f"Training epoch {epoch}:")
        train(loader, models_from, models_to, loss_fn, optimizers_from, optimizers_to, device)


def export_models(models_from, models_to):
    """Export pytorch models as serialized files"""

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    for i, piece in enumerate(chess_pieces):
        torch.save(models_from[piece].state_dict(), f"model_from_{i}.pt")
        torch.save(models_to[piece].state_dict(), f"model_to_{i}.pt")


def train_on_games_and_export(pgn_file, epochs):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    models_from = {}
    models_to = {}

    chess_pieces = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)

    for piece in chess_pieces:
        models_from[piece] = StackedConvolve().to(device)
        models_to[piece] = StackedConvolve().to(device)

    train_on_games(pgn_file, models_from, models_to, epochs, device)
    export_models(models_from, models_to)
