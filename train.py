from model import StackedConvolve
import torch
from torch import nn
from torch.utils.data import DataLoader

import chess.pgn

class ChessDataset(torch.utils.data.Dataset):
    """Torch dataset containing (piece, board state, from square, to square)"""

    best_moves: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(self, move_pairs):
        self.best_moves = []

        # Extract data from board state and moves
        for board, move in move_pairs:
            self.best_moves.append((
                board.piece_at(move.from_square).piece_type,
                chess_to_tensor(board),
                position_to_tensor(move.from_square),
                position_to_tensor(move.to_square)))

    def __len__(self):
        return len(best_moves)

    def __getitem__(self, idx):
        return best_moves[idx]


def load_data(pgn_file):
    """Loads a PGN file and stores each move in (board state, move) pairs"""

    pgn = open(pgn_file)

    game = chess.pgn.read_game(pgn)
    game_tuples = []

    # Iterate as long as a game can be found in the pgn file
    while game is not None:
        # Start each game with a standard board layout
        board.reset()

        # Simulate each move and add it to the list
        for move in game.mainline(game.game()):
            game_tuples.append((board, move))
            board.push(move)
            board.apply_mirror()

        # Attempt to read another game
        game = chess.pgn.read_game(pgn)

    return game_tuples


def train(dataloader, models_from, models_to, loss_fn, optimizers_from, optimizers_to, device):
    """Trains two lists of models based on a dataset using pre-specified optimizers"""

    size = len(dataloader.dataset)

    # Set all models to training mode
    map(lambda x: x.train(), models_from)
    map(lambda x: x.train(), models.to)

    for batch, (piece, X, y, z) in enumerate(dataloader):
        X, y, z = X.to(device), y.to(device), z.to(device)

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

    game_pairs = load_data(pgn_file)
    dataset = ChessDataest(game_pairs)
    loader = DataLoader(dataset)

    # Define loss function and create list of optimizers for each model in lists
    loss_fn = nn.CrossEntropyLoss()
    optimizers_from = map(lambda x: torch.optim.SGD(x.parameters(), lr=1e-3), models_from)
    optimizers_to = map(lambda x: torch.optim.SGD(x.parameters(), lr=1e-3), models_to)

    # Iteratively train over epochs
    for epoch in range(epochs):
        print(f"Training epoch {epoch}:")
        train(loader, models_from, models_to, loss_fn, optimizers_from, optimizers_to, device)
