import itertools
import sys
import time

import argparse
import numpy as np

import QLearning
import StickGame
from utils import ask_for_player_input


def two_players():
    """Launches a two-player game."""
    # Initialize game
    game = StickGame.Game()

    # Initialize first player
    players = ["A", "B"]
    players_iterator = itertools.cycle(players)
    player = next(players_iterator)

    # play
    while not game.is_finished():
        game.print_state()
        played = False
        while not played:
            line_input, sticks_input = ask_for_player_input(player)
            played = game.play(line_input, sticks_input)
        player = next(players_iterator)
    sys.stdout.write(f"Player {player} won!\n")


def against_ai(arguments):
    """Launches a game against a trained ai, using its name and its number of training_epochs."""
    ai_name, training_epochs = arguments
    training_epochs = int(training_epochs)

    # Initialize ai
    try:
        agent = QLearning.Agent(learning=False)
        agent.load(ai_name, training_epochs)
    except FileNotFoundError:
        sys.stdout.write("No cached agent found with these parameters. Please train it first with the following command:\n")
        sys.stdout.write(f"> python main.py --train-ai  {ai_name} {training_epochs}\n")
        sys.exit(1)

    # Initialize game
    game = StickGame.Game()

    # Initialize first player
    players = ["You", ai_name + " (AI)"]
    players_iterator = itertools.cycle(players)
    player = next(players_iterator)
    you_start = input("Do you want to start? [Y/n] -> ")
    if you_start == "n":
        player = next(players_iterator)

    # play
    while not game.is_finished():
        game.print_state()
        played = False
        if player == "You":
            while not played:
                line_input, sticks_input = ask_for_player_input(player)
                played = game.play(line_input, sticks_input)
            player = next(players_iterator)
        else:
            line, sticks, _, _ = agent.td_learning(game)
            sys.stdout.write(f"\n{ai_name} + (AI) takes {sticks} sticks in line {line}\n\n\n")
            player = next(players_iterator)
    sys.stdout.write(f"{player} won!\n")


def train_ai(arguments):
    """Trains an agent for training_epochs epochs and stores it using its name and training_epcochs."""
    ai_name, training_epochs = arguments
    training_epochs = int(training_epochs)

    epochs_rewards = []
    epochs_fails = []
    start_time = time.time()
    update_freq = training_epochs // 10

    # Initialize ai
    learning_agent = QLearning.Agent(learning=True)
    adversarial_agent = QLearning.Agent(learning=False)
    for epoch in range(training_epochs):
        epoch_reward = 0
        epoch_fail = 0
        learning_agent.increment_training_epochs()
        if not (epoch + 1) % update_freq:
            learning_agent.save(ai_name)
            sys.stdout.write(f"epoch {epoch+1} out of {training_epochs}\n")
            adversarial_agent.copy(learning_agent, learning=False)

        # Initialize game
        game = StickGame.Game()

        # Initialize first player
        players = [learning_agent, adversarial_agent]
        players_iterator = itertools.cycle(players)
        player = next(players_iterator)
        if np.random.random() <= 0.5:
            # With probability .5, make the learning agent start or not
            player = next(players_iterator)

        # play
        while not game.is_finished():
            _, _, action_reward, action_fails = player.td_learning(game)
            player = next(players_iterator)
            if player.learning:
                epoch_reward += action_reward
                epoch_fail += action_fails

        epochs_rewards.append(epoch_reward)
        epochs_fails.append(epoch_fail)

    sys.stdout.write(
        f"Training took {round(time.time()-start_time, 2)}s for {training_epochs} epochs.\n"
    )
    return epochs_rewards, epochs_fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--two-players', action='store_true', help="Launches a two-player game.")
    parser.add_argument('--train-ai', nargs=2, metavar=("ai_name", "epochs"), help="Trains an agent for training_epochs epochs and stores it using its name and training_epcochs.")
    parser.add_argument('--against-ai', nargs=2, metavar=("ai_name", "epochs"), help="Launches a game against a trained ai, using its name and its number of training_epochs.")

    args = parser.parse_args()

    if args.two_players:
        two_players()
    elif arguments := args.against_ai:
        against_ai(arguments)
    elif arguments := args.train_ai:
        train_ai(arguments)
    else:
        print("Wrong arguments...")