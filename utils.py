import os
import sys

def check_create_folder(relative_path_to_folder):
    if not os.path.exists(relative_path_to_folder):
        os.mkdir(relative_path_to_folder)

def ask_for_player_input(player):
    sys.stdout.write("\n\n")
    sys.stdout.write(f"Player {player} it is your turn \n")
    accepted_input = False
    while not accepted_input:
        sticks_input = input("How many sticks do you want to get rid of? -> ")
        line_input = input("On what line? -> ")
        try:
            line_input = int(line_input)
            sticks_input = int(sticks_input)
            accepted_input = True
        except ValueError:
            sys.stdout.write("Please enter integer(s)\n")
    return line_input, sticks_input