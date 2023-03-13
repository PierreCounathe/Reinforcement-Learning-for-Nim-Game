# Reinforcement-Learning-for-Nim-Game
The Nim game is a mathematical strategy game in which two players take turns removing sticks from distinct heaps. The player removing the last sticks loses. I build a version of the game that can be played in the terminal, and then an Agent class that plays the game. This Agent implements the Q-Learning algorithm. 

# Play the game
1. Clone the repository:
```
git clone https://github.com/PierreCounathe/Reinforcement-Learning-for-Nim-Game
```
2. Train the Agent and specify its name and number of `training_epochs`:
```
python3 main.py -train-ai ai_name 100000
```
3. Play against another human or against a trained Agent:
```
python3 main.py -two-players
```
```
python3 main.py -against-ai ai_name 100000
