import numpy as np
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from env.network_env import DigitalTwin5GEnv
from agent.dqn_agent import DQNAgent

# Load trained models
demand_model = load_model("demandPredictionTanh2.h5")
classification_model = load_model("ann_model.h5")

# Setup environment and agent
env = DigitalTwin5GEnv()
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)

# Run simulation
episodes = 50
for e in range(episodes):
    state = env.reset()  # shape: (60, 11)
    total_reward = 0
    done = False
    step = 0

    print(f"\n🔁 Starting Episode {e + 1}/{episodes}")
    print(f"📊 Initial State Shape: {state.shape}, Type: {type(state)}")

    while not done:
        # Predict demand using raw state shape (60, 11)
        demand_pred = demand_model.predict(np.expand_dims(state, axis=0), verbose=0).flatten()[0]

        # For traffic class prediction, flatten state and take first 3 features
        state_flat = state.flatten()
        input_for_classification = state_flat[:3].reshape(1, 3)
        slice_pred = np.argmax(classification_model.predict(input_for_classification, verbose=0), axis=1)[0]

        # Select action using RL agent
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Train agent
        agent.train(state, action, reward, next_state, done)

        print(f"🧠 Step {step + 1} | Demand: {demand_pred:.2f} | Traffic Type: {slice_pred} | Action: {action} | Reward: {reward:.2f} | Done: {done}")

        total_reward += reward
        state = next_state
        step += 1

    print(f"✅ Episode {e + 1} finished with Total Reward: {total_reward:.2f}")

