# Blackjack Reinforcement Learning Agent

A Q-learning implementation to solve the Blackjack environment using the Gymnasium library. This project demonstrates core concepts of reinforcement learning (RL) and provides a practical example of training an RL agent.

## Main Reinforcement Learning Components

1. **Environment**: `gym.make('Blackjack-v1')`  
   - Simulates Blackjack rules with discrete state/action spaces
   - State space: (player_sum, dealer_card, usable_ace)
   - Action space: [Hit (0), Stand (1)]

2. **Agent**: Q-learning algorithm with:
   - **Q-table**: 4D tensor `(32, 11, 2, 2)` tracking state-action values
   - **Exploration-Exploitation**: ε-greedy strategy with decay
   - **Learning Mechanism**: Temporal Difference updates

3. **Reward System**:
   - Win: +1
   - Loss: -1
   - Draw: 0

4. **Learning Parameters**:
   - α (learning rate) = 0.2
   - γ (discount factor) = 1.0
   - ε-decay = 0.99999
   - Training episodes: 500,000


## Code Structure & Key Components

### Q-learning
```python
# Q-table initialization with basic strategy
Q = np.zeros((32, 11, 2, 2))  # Dimensions: (player_sum, dealer_card, usable_ace, action)

# ε-Greedy Action Selection
def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return np.argmax(Q[state[0], state[1], state[2], :])  # Exploitation

# Q-learning Update Rule
current_q = Q[state[0], state[1], state[2], action]
target = reward + gamma * np.max(Q[next_state[...]])
Q[state[...], action] += alpha * (target - current_q)

```

### Training Process
Initializes Q-table with basic strategy (hit below 17, stand otherwise)

Iterates through 500,000 episodes:

Uses ε-greedy policy for action selection

Updates Q-values using temporal difference learning

Gradually reduces exploration (ε decay)

Performs periodic evaluations:

Every 10,000 episodes tests policy on 1,000 games

Tracks win/loss/draw rates and average reward

### Visualization feature

```python

def visualize_games(num_games=5, delay=1.0):
    '''Demonstrates agent's decisions with rendering'''
    # Uses human-readable output showing:
    # - Current state (player/dealer cards, usable ace)
    # - Chosen action
    # - Game outcome
```

### Expected Output

Episode: 490000\
Epsilon: 0.0100\
Win Rate: 43.2% | Loss Rate: 49.1% | Draw Rate: 7.7%\
Average Reward: -0.0590

Final Evaluation (10000 games):\
Average Reward: -0.0423


## Results
Final evaluation after training typically shows:

Win rate: ~40-44%

Loss rate: ~48-51%

Draw rate: ~8-10%


### Run the code

Install requirements:
```sh
pip install -r requirements.txt
```

Run main script:
```sh
python blackjack_gym.py
```

### References

https://gymnasium.farama.org/environments/toy_text/blackjack/