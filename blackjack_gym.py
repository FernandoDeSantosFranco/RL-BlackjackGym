import gymnasium as gym
import numpy as np
import time

# Environment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Q-table initialization with basic strategy
Q = np.zeros((32, 11, 2, 2))  # (player_sum, dealer_card, usable_ace, action)
for player_sum in range(4, 32):
    for dealer_card in range(1, 11):
        for usable_ace in [0, 1]:
            if player_sum < 17:
                Q[player_sum, dealer_card, usable_ace, 0] = 1  # Hit
            else:
                Q[player_sum, dealer_card, usable_ace, 1] = 1  # Stand

# Hyperparameters
alpha = 0.2    # Learning rate
gamma = 1.0    # Discount factor (1.0 for terminal games)
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.99999
epsilon_min = 0.01
num_episodes = 500_000

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state[0], state[1], state[2], :])  # Exploit

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning update
        current_q = Q[state[0], state[1], state[2], action]
        target = reward if done else reward + gamma * np.max(Q[next_state[0], next_state[1], next_state[2], :])
        Q[state[0], state[1], state[2], action] += alpha * (target - current_q)
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Evaluation every 10,000 episodes
    if episode % 10_000 == 0:
        wins, losses, draws = 0, 0, 0
        eval_rewards = []
        
        for _ in range(1000):
            state, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = np.argmax(Q[state[0], state[1], state[2], :])
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

        print(f"""
        Episode: {episode}
        Epsilon: {epsilon:.4f}
        Win Rate: {wins/10}% | Loss Rate: {losses/10}% | Draw Rate: {draws/10}%
        Average Reward: {np.mean(eval_rewards):.4f}
        """)

# Final evaluation
total_reward = 0
num_eval_episodes = 10_000
for _ in range(num_eval_episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state[0], state[1], state[2], :])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    total_reward += reward

print(f"\nFinal Evaluation ({num_eval_episodes} games):")
print(f"Average Reward: {total_reward/num_eval_episodes:.4f}")

# Visualization function
def visualize_games(num_games=5, delay=1.0):
    """Show complete games with rendering and decision details"""
    demo_env = gym.make('Blackjack-v1', render_mode='human', natural=False, sab=False)
    
    for game in range(num_games):
        print(f"\n=== Game {game+1}/{num_games} ===")
        state, info = demo_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = np.argmax(Q[state[0], state[1], state[2], :])
            next_state, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated
            
            # Print decision
            print(f"Player: {state[0]} | Dealer: {state[1]} | Usable Ace: {bool(state[2])}")
            print(f"Chose to {'Hit' if action == 0 else 'Stand'} | Current Reward: {reward}")
            
            demo_env.render()
            time.sleep(delay)
            state = next_state
            total_reward += reward
        
        # Final result
        result = "Win" if reward == 1 else "Loss" if reward == -1 else "Draw"
        print(f"Final Result: {result} | Total Reward: {total_reward}")
        time.sleep(2)  # Pause between games
    
    demo_env.close()

# Visualization with 5 games
visualize_games(num_games=5, delay=1.5)

env.close()