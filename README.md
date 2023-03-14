# Reinforcement-Learning-for-Nim-Game
The Nim game is a mathematical strategy game in which two players take turns removing sticks from distinct heaps. The player who removes the last stick loses. As a game in which states are fully observable and with much reduced state and action spaces compared to chess, it is the perfect game to code a Q-Learning RL agent from scratch.

<p align="center">
  <img src="https://i.pinimg.com/originals/8d/db/49/8ddb49378353a8ee860e081a96de8d4e.jpg"
       alt="Nim Game (Last Year at Marienbad (1961))"/>
 <p/>
     
## Play the game
1. Clone the repository:
```
git clone https://github.com/PierreCounathe/Reinforcement-Learning-for-Nim-Game
```
```
cd Reinforcement-Learning-for-Nim-Game
```
2. Train the Agent and specify its name and number of `training_epochs`:
```
python3 main.py -train-ai ai_name 5000
```
3. Play against another human or against a trained Agent:
```
python3 main.py -two-players
```
```
python3 main.py -against-ai ai_name 5000
