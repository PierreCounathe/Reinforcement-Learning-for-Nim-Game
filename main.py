import itertools
import sys
import time

import argparse
import numpy as np

import QLearning
import StickGame


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
            sys.stdout.write(f"Player {player} it is your turn \n")
            line_input = int(input("What line do you want to play? -> "))
            sticks_input = int(input("How many sticks do you want to play? -> "))
            played = game.play(line_input, sticks_input)
        player = next(players_iterator)
    sys.stdout.write(f"Player {player} won!\n")


def against_ai(other_args):
    """Launches a game against a trained ai, using its name and its number of training_epochs."""
    # Initialize game
    game = StickGame.Game()

    # Initialize first player
    players = ["You", "AI"]
    players_iterator = itertools.cycle(players)
    player = next(players_iterator)
    you_start = input("Do you want to start? [Y/n] -> ")
    if you_start == "n":
        player = next(players_iterator)

    # Initialize ai
    agent = QLearning.Agent(learning=False)
    if len(other_args) == 2:
        agent.load(other_args[0], other_args[1])

    # play
    while not game.is_finished():
        game.print_state()
        played = False
        if player == "You":
            while not played:
                sys.stdout.write(f"Player {player} it is your turn \n")
                line_input = int(input("What line do you want to play? -> "))
                sticks_input = int(input("How many sticks do you want to play? -> "))
                played = game.play(line_input, sticks_input)
            player = next(players_iterator)
        else:
            line, sticks, _, _ = agent.td_learning(game)
            sys.stdout.write(f"AI takes {sticks} sticks in line {line}\n")
            player = next(players_iterator)
    sys.stdout.write(f"{player} won!\n")


def train_ai(other_args):
    """Trains an agent for training_epochs epochs and stores it using its name and training_epcochs."""
    epochs_rewards = []
    epochs_fails = []
    start_time = time.time()
    update_freq = int(other_args[1]) // 10

    # Initialize ai
    learning_agent = QLearning.Agent(learning=True)
    adversarial_agent = QLearning.Agent(learning=False)
    for epoch in range(int(other_args[1])):
        epoch_reward = 0
        epoch_fail = 0
        learning_agent.increment_training_epochs()
        if not (epoch + 1) % update_freq:
            learning_agent.save(other_args[0])
            sys.stdout.write(f"epoch {epoch+1} out of {int(other_args[1])}\n")
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
        f"Training took {round(time.time()-start_time, 2)}s for {int(other_args[1])} epochs.\n"
    )
    return epochs_rewards, epochs_fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Two players
    two_players_parser = subparsers.add_parser('two-players')
    
    # Against AI
    against_ai_parser = subparsers.add_parser('against-ai')
    against_ai_parser.add_argument('ai_name', default='NimGameAI', required=False, help="Name of the AI")
    against_ai_parser.add_argument('training_epochs', type=int, default=5000, required=False, help="Number of training epochs")

    # Train AI
    train_ai_parser = subparsers.add_parser('train-ai')
    train_ai_parser.add_argument('ai_name', required=True, help="Name of the AI")
    train_ai_parser.add_argument('training_epochs', type=int, required=True, help="Number of training epochs")

    args = parser.parse_args()

    if args.command == 'two-players':
        two_players()
    elif args.command == 'against-ai':
        against_ai(args)
    elif args.command == 'train-ai':
        train_ai(args)
    else:
        print("Wrong arguments...")