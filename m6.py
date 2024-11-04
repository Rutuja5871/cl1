import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Maze Environment
maze = np.array([
    [0, 0, 0, 1, 0],  # 1 indicates an obstacle
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 0, 2]   # 2 indicates the goal
])

# Maze parameters
n_rows, n_cols = maze.shape
start_state = (0, 0)
goal_state = (4, 4)

# Step 2: Define Q-Learning Parameters
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((n_rows, n_cols, len(actions)))  # Q-values for each (state, action)
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.9    # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
min_epsilon = 0.1  # Minimum value of epsilon
n_episodes = 1000  # Number of episodes for training

# Helper function to choose an action using epsilon-greedy strategy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(actions))  # Explore: random action
    else:
        return np.argmax(q_table[state])  # Exploit: best action from Q-table

# Helper function to get the next state based on the action taken
def get_next_state(state, action):
    row, col = state
    
    # Calculate new position based on action
    if action == 0 and row > 0:  # Move up
        row -= 1
    elif action == 1 and row < n_rows - 1:  # Move down
        row += 1
    elif action == 2 and col > 0:  # Move left
        col -= 1
    elif action == 3 and col < n_cols - 1:  # Move right
        col += 1
    
    # Check if the new position is an obstacle
    if maze[row, col] == 1:
        return state  # If it is an obstacle, stay in the current state
    return (row, col)

# Helper function to get the reward for a given state
def get_reward(state):
    if state == goal_state:
        return 10  # High reward for reaching the goal
    elif maze[state] == 1:
        return -10  # Penalty for hitting an obstacle
    else:
        return -1  # Small penalty for each step to encourage faster solutions

# Step 3: Train the Q-Learning Agent
for episode in range(n_episodes):
    state = start_state
    total_reward = 0

    while state != goal_state:
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Update Q-Table using the Q-Learning formula
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_reward += reward

    # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

print("\nTraining completed!")

# Step 4: Visualize the Optimal Path Found by the Agent
def visualize_optimal_path():
    path = np.zeros_like(maze, dtype=str)
    path[:] = ' '

    state = start_state
    max_steps = 50  # Set a max limit to prevent infinite loops
    steps = 0

    while state != goal_state and steps < max_steps:
        action = np.argmax(q_table[state])
        next_state = get_next_state(state, action)

        # If the agent enters a loop or gets stuck, exit
        if path[state] == 'A':  # If already visited
            print("\nNo route found. The agent is stuck in a loop.")
            return

        path[state] = 'A'  # Mark the agent's path
        state = next_state
        steps += 1
        print(path)

    # Check if goal was reached
    if state == goal_state:
        path[goal_state] = 'G'  # Mark the goal
        print("\nOptimal Path Found by the Agent:")
        print(path)
    else:
        print("\nNo route found. The agent did not reach the goal.")

visualize_optimal_path()


#-----------------------------------------------------------------------------------


import numpy as np

# Define the maze (0 = free space, 1 = obstacle, 2 = goal)
maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 2]
])

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table
num_rows, num_cols = maze.shape
num_states = num_rows * num_cols
num_actions = 4  # up, down, left, right
Q = np.zeros((num_states, num_actions))

# Action mappings
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

def get_state(row, col):
    return row * num_cols + col

def get_next_state_and_reward(state, action):
    row, col = divmod(state, num_cols)
    dr, dc = actions[action]
    new_row, new_col = row + dr, col + dc

    if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
        new_state = get_state(new_row, new_col)
        if maze[new_row, new_col] == 1:
            return state, -1
        elif maze[new_row, new_col] == 2:
            return new_state, 10
        return new_state, -1
    return state, -1

# Training with Q-learning
for _ in range(num_episodes):
    state = get_state(0, 0)  # Start at (0, 0)
    while True:
        action = np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(Q[state])
        new_state, reward = get_next_state_and_reward(state, action)
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[new_state]) - Q[state, action])
        state = new_state
        if reward == 10:
            break

# Print the optimal path
state = get_state(0, 0)
goal_state = get_state(4, 4)
path = []

while state != goal_state:
    path.append(divmod(state, num_cols))
    action = np.argmax(Q[state])
    state, _ = get_next_state_and_reward(state, action)

path.append((4, 4))  # Goal position
print("Optimal Path:", " -> ".join(map(str, path)))

