from torch import manual_seed, optim
from source.agent import Agent
from source.critic import Critic
from source.policy import Policy
from utils.miscellaneous import load_best_model


# Settings and hyperparameters
config = {
    "num_train_episodes": 20000,
    "num_test_episodes": 100,
    "num_environments": 6,
    "episode_length": 32,
    "learning_rate_policy": 1e-3,
    "learning_rate_critic": 1e-4,
    "weight_decay_critic": 1e-5,
    "integration_method": 'rk4',
}


def main():
    # Set random seeds for reproducibility
    manual_seed(42)

    # Set up environment, policy, critic, and agent
    policy = Policy()
    critic = Critic()
    # load_best_model("./models/shac_last_model.pt", policy, critic)

    policy_optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate_policy"])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config["learning_rate_critic"], weight_decay=config["weight_decay_critic"])
    agent = Agent(config["num_environments"], config["episode_length"], policy, critic, policy_optimizer, critic_optimizer)

    # Train and test the policy
    manual_seed(42)
    # untrained_cumulative_rewards = agent.test_policy(config["num_test_episodes"])
    # untrained_mean_score = untrained_cumulative_rewards.mean()

    agent.train_policy(config["num_train_episodes"])

    # manual_seed(42)
    # trained_cumulative_rewards = agent.test_policy(config["num_test_episodes"])
    # trained_mean_score = trained_cumulative_rewards.mean()

    # print(f'Mean score before training: {untrained_mean_score}')
    # print(f'Mean score after training:  {trained_mean_score}')
    print('done')


if __name__ == "__main__":
    main()
