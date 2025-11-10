# Griddle-Me-This
A neural network-based chess evaluation library/program

# Demo

Demo coming soon...

# Running
In the "Releases" section, a set of example models has been provided. You should place them in the same flat directory as `game.py`. Then, to start the interactive gameplay, you can run
```
python3 game.py
```
The game will expect you to enter your moves in algebraic notation, with the starting square and the ending square. To exit, you can type Ctrl-C.

## Training
Currently, training can only be performed by giving the model a dataset containing games in .pgn format. In the REPL, you can call
```
>>> from train import *
>>> train_iteratively_on_games_and_export(<pgn_file>, <epochs>)
```
Where `<pgn_file>` represents whichever dataset you are using and `<epochs>` is the number of iterations over that dataset you want to perform.

# Design
This model was based on the ideas proposed in the [ConvChess](https://cs231n.stanford.edu/reports/2015/pdfs/ConvChess.pdf) paper with modifications.

The model takes the board input as 12 8x8 grids (represented by a 12x8x8 tensor), with a bitboard-style binary encoding for where each piece is located. The output is given through 2 tensors: the first gives a probability for each square being the starting place of the moved piece, and the second gives the probability for each square being the ending square of a given piece. To yield the best move, these probabilities are multiplied.

The selection of parameters and layers was Ad-hoc but very loosely based on ConvChess. The design was strongly influenced by the paper's design.

Right now, there are 4 layers:
1. Convolution + Linear + Leaky Relu (separated by channel)
2. Convolution + Linear + Leaky Relu (interacts between all 12 channels)
3. Convolution + Linear + Leaky Relu (reduces channels to 1 channel)
4. Convolution + Linear + Leaky Relu (acts on the one remaining channel)

Then, the result is flattened and run through a softmax layer to yield probabilities.
