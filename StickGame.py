import sys

import numpy as np


class Game:
    def __init__(self) -> None:
        """Initializes the stick heaps."""
        self.state = np.array([1, 3, 5, 7])

    def reset(self) -> None:
        """Resets the stick heaps."""
        self.state = np.array([1, 3, 5, 7])

    def is_valid_line(self, line: int) -> bool:
        """Check if a line can possibly be played."""
        return 0 <= line <= 3

    def is_valid_sticks(self, line: int, sticks: int) -> bool:
        """Check if the number of sticks to be played can be played."""
        return (self.state[line] >= sticks) and sticks >= 1

    def is_valid_play(self, line: int, sticks: int) -> bool:
        """Checks the validity of the line, sticks combination."""
        return self.is_valid_line(line) and self.is_valid_sticks(line, sticks)

    def play(self, line: int, sticks: int) -> bool:
        """Updates the game state if the play is valid, and if not, writes that the play is not valid."""
        if not self.is_valid_play(line, sticks):
            sys.stdout.write(f"You cannot make this play\n")
            return False
        self.state[line] -= sticks
        return True

    def print_state(self) -> None:
        """Writes the game state in the terminal."""
        sys.stdout.write("Current state is: \n")
        for line_index, number_of_sticks in enumerate(self.state):
            sticks_to_print = "|" * number_of_sticks
            sys.stdout.write(f"{line_index}: {sticks_to_print} \n")

    def get_state_int_encoded(self) -> int:
        """Returns the Integer encoding of the current game state."""
        string = ""
        for sticks in self.state:
            string += str(sticks)
        return int(string)

    def is_finished(self) -> bool:
        """Returns True if the game is over."""
        return sum(self.state) == 0
