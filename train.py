from model import StackedConvolve
import torch
from torch import nn
from torch.utils.data import DataLoader

import chess.pgn

class ChessDataset(torch.utils.data.Dataset):
    best_moves: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(self, move_pairs):
        self.best_moves = []
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
    pgn = open(pgn_file)

    game = chess.pgn.read_game(pgn)
    game_tuples = []

    while game is not None:
        board.reset()
        for move in game.mainline(game.game()):
            game_tuples.append((board, move))
            board.push(move)
            board.apply_mirror()

        game = chess.pgn.read_game(pgn)

    return game_tuples


def train(dataloader, models_from, models_to, loss_fn, optimizers_from, optimizers_to):
    size = len(dataloader.dataset)

    map(lambda x: x.train(), models_from)
    map(lambda x: x.train(), models.to)

    for batch, (piece, X, y, z) in enumerate(dataloader):
        X, y, z = X.to(device), y.to(device), z.to(device)

        pred_from = models_from[piece](X)
        loss_from = loss_fn(pred_from, y)

        loss_from.backward()
        optimizers_from[piece].step()
        optimizers_from[piece].zero_grad()

        pred_to = models_to[piece](X)
        loss_to = loss_fn(pred_to, z)

        loss_to.backward()
        optimizers_to[piece].step()
        optimizers_to[piece].zero_grad()


def train_on_games(pgn_file, models_from, models_to, epochs):
    game_pairs = load_data(pgn_file)
    dataset = ChessDataest(game_pairs)
    loader = DataLoader(dataset)

    loss_fn = nn.CrossEntropyLoss()
    optimizers_from = map(lambda x: torch.optim.SGD(x.parameters(), lr=1e-3), models_from)
    optimizers_to = map(lambda x: torch.optim.SGD(x.parameters(), lr=1e-3), models_to)

    for epoch in range(epochs):
        print(f"Training epoch {epoch}:")
        train(loader, models_from, models_to, loss_fn, optimizers_from, optimizers_to)
