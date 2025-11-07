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


def train(dataloader, model_from, model_to, loss_fn, optimizer_from, optimizer_to):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y, z) in enumerate(dataloader):
        X, y, z = X.to(device), y.to(device), z.to(device)

        pred_from = model(X)
        loss_from = loss_fn(pred, y)

        loss_from.backward()
        optimizer_from.step()
        optimizer_from.zero_grad()

        pred_to = model(X)
        loss_to = loss_fn(pred, z)

        loss_to.backward()
        optimizer_to.step()
        optimizer_to.zero_grad()


def train_on_games(pgn_file, model_from, model_to, epochs):
    game_pairs = load_data(pgn_file)
    dataset = ChessDataest(game_pairs)
    loader = DataLoader(dataset)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_from = torch.optim.SGD(model_from.parameters(), lr=1e-3)
    optimizer_to = torch.optim.SGD(model_to.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Training epoch {epoch}:")
        train(loader, model_from, model_to, loss_fn, optimizer_from, optimizer_to)
