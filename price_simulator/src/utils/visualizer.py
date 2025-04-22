from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate(data: List, desired_length: int = 100) -> pd.Series:
    """Convert data to pandas series and group data to desired length."""
    if len(data) > desired_length:
        df = pd.Series(data, index=np.floor(np.linspace(1, desired_length + 1, len(data), endpoint=False)))
        average_df = df.groupby(df.index).mean()
        average_df.index = average_df.index * len(df) / desired_length
        return average_df
    else:
        return pd.Series(data)


def create_subplot(yy: List, label: str, ax=None, agg: bool = False):
    palette = plt.get_cmap("Set1")
    ax = ax or plt.gca()
    num = 0
    for y in yy:
        num += 1
        if agg:
            ax.plot(aggregate(y), marker="", color=palette(num), linewidth=1, alpha=0.9, label="Agent {}".format(num))
        else:
            ax.plot(y, marker="", color=palette(num), linewidth=1, alpha=0.9, label="Agent {}".format(num))

    ax.legend(loc=2, ncol=2)
    ax.set(xlabel="Period", ylabel=label)

    return ax

def visualize_results(env, showAgent1: bool = True, showAgent2: bool = True):
    """Visualize loss history, price history, and reward history for the given environment."""
   
    # Plot loss history if it exists
    if hasattr(env.agents[0], "loss_history"):
        plt.figure(figsize=(12, 6))
        plt.plot(env.agents[0].loss_history, label="Agent 1 Loss")
        plt.title("Loss History")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("Loss history attribute not found for Agent 1.")

    # Analyze and visualize results
    price_history_agent_1 = [price[0] for price in env.price_history]
    price_history_agent_2 = [price[1] for price in env.price_history]

    # Plot price history with monopoly and Nash prices
    plt.figure(figsize=(12, 6))
    if showAgent1:
        plt.plot(price_history_agent_1, label="Agent 1")
    if showAgent2:
        plt.plot(price_history_agent_2, label="Agent 2")
    
    # Add monopoly and Nash prices if they exist in the environment
    if hasattr(env, "monopoly_prices"):
        plt.axhline(y=env.monopoly_prices[0], color="red", linestyle="--", label=f"Monopoly Price Agent {1}")
    if hasattr(env, "nash_prices"):
        plt.axhline(y=env.nash_prices[0], color="green", linestyle="--", label=f"Nash Price Agent {1}")
    
    plt.title("Price History: Agent 1 vs Agent 2")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Plot reward history
    reward_history_agent_1 = [reward[0] for reward in env.reward_history]
    reward_history_agent_2 = [reward[1] for reward in env.reward_history]
    plt.figure(figsize=(12, 6))
    if showAgent1:
        plt.plot(reward_history_agent_1, label="Agent 1")
    if showAgent2:
        plt.plot(reward_history_agent_2, label="Agent 2")
    plt.title("Reward History: Agent 1 vs Agent 2")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()