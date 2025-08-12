import time
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_value_
import torch.nn.functional as F
from matplotlib import pyplot as plt
from source.critic import Critic
from source.environment import Environment, EnvironmentParameters
from source.memory import Memory
from source.policy import Policy
#from utils.miscellaneous import save_best_model


class Agent:
    def __init__(self, num_environments: int, episode_length: int,
                 policy: Policy, critic: Critic, policy_optimizer, critic_optimizer,
                 gamma=0.99, lambda_=0.95, alpha=0.995):
        self.policy = policy
        self.critic = critic
        self.target_critic = Critic()
        self.target_critic.load_state_dict(critic.state_dict())
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha = alpha  # alpha lr i.e. soft update of the target critic network 

        self.GAMMA  = torch.tensor(gamma)
        self.LAMBDA = torch.tensor(lambda_)

        self.episode_length   = episode_length
        self.num_environments = num_environments
        self.initialize_environments()

        tmpObs = self.environments[0].get_observation()
        len_obs     = len(tmpObs)
        len_action  = len(self.policy(tmpObs)[0])
        self.memory = Memory(self.num_environments, self.episode_length, len_obs, len_action)
        self.initialize_figure()
        self.initialize_reward_plot()
        
        self.wall_clock_times = []
        self.total_time_elapsed = 0
        self.best_avg_reward = -float('inf')

    # initialize multiple instances of an Env. class.
    # Each env. has its own unique initial step but shares the same params and price data.
    def initialize_environments(self):
        env_params = EnvironmentParameters()

        prices, max_step   = Environment.load_prices(env_params.price_prediction_horizon)
        self.initial_steps = torch.randint(max_step, size=(self.num_environments,))
        self.environments  = [Environment(env_params, prices, i, max_step) for i in self.initial_steps]

    def initialize_figure(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(5, 1)

        self.lines     = [None] * 8
        self.lines[0], = self.ax[0].plot([], [], 'k')
        self.lines[1], = self.ax[1].plot([], [], 'k')
        self.lines[2], = self.ax[2].plot([], [], 'k-')
        self.lines[3], = self.ax[2].plot([], [], 'r-')
        self.lines[4], = self.ax[3].plot([], [], 'k-')
        self.lines[5], = self.ax[3].plot([], [], 'b-')
        self.lines[6], = self.ax[4].plot([], [], 'k-')
        self.lines[7], = self.ax[4].plot([], [], 'b-')
        
        self.ax[0].set_ylabel('Policy loss')
        self.ax[1].set_ylabel('Critic loss')
        self.ax[2].set_ylabel('Penalty')
        self.ax[3].set_ylabel('Reward')
        self.ax[4].set_ylabel('Reward')
        self.ax[3].set_xlabel('Iteration')
        self.ax[4].set_xlabel('Wall-clock time')
        self.ax[3].xaxis.label.set_visible(True)
        self.ax[4].xaxis.label.set_visible(True)
        
        self.ax[0].set_yscale('log')
        self.ax[1].set_yscale('log')
        self.ax[2].set_yscale('log')
        self.ax[3].set_yscale('log')
        self.ax[4].set_yscale('log')

        self.x_data             = []
        self.policy_loss        = []
        self.critic_loss        = []
        self.mean_penalty_loss  = []
        self.max_penalty_loss   = []
        self.mean_reward        = []
        self.max_reward         = []
        self.iteration          = 0

    def update_plot(self, policy_loss, critic_loss, start_time):
        penalties = self.memory.penalties.detach().numpy()
        rewards = self.memory.rewards.detach().numpy()

        elapsed_time = time.time() - start_time
        self.x_data.append(self.iteration)
        self.total_time_elapsed += elapsed_time
        self.wall_clock_times.append(self.total_time_elapsed)

        self.policy_loss.append(policy_loss)
        self.critic_loss.append(critic_loss)
        self.mean_penalty_loss.append(np.mean(penalties))
        self.max_penalty_loss.append(np.max(penalties))
        self.mean_reward.append(np.mean(rewards))
        self.max_reward.append(np.max(rewards))
        self.iteration += 1

        self.lines[0].set_data(self.x_data, self.policy_loss)
        self.lines[1].set_data(self.x_data, self.critic_loss)
        self.lines[2].set_data(self.x_data, self.mean_penalty_loss)
        self.lines[3].set_data(self.x_data, self.max_penalty_loss)
        self.lines[4].set_data(self.x_data, self.mean_reward)
        self.lines[5].set_data(self.x_data, self.max_reward)
        self.lines[6].set_data(self.wall_clock_times, self.mean_reward)
        self.lines[7].set_data(self.wall_clock_times, self.max_reward)

        [self.ax[i].relim() for i in range(5)]
        [self.ax[i].autoscale_view() for i in range(5)]

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def initialize_reward_plot(self):
        plt.ion()
        self.fig_reward, (self.ax_time, self.ax_steps) = plt.subplots(1, 2, figsize=(12, 6))
        
        self.x_data             = []
        self.mean_reward        = []
        self.iteration          = 0

        # Plot for mean reward vs wall clock time
        self.line_mean_reward_time, = self.ax_time.plot([], [], 'b-')
        self.ax_time.set_ylabel('Average reward')
        self.ax_time.set_xlabel('Wall-clock time')
        self.ax_time.xaxis.label.set_visible(True)
        self.ax_time.set_yscale('log')
        
        # Plot for mean reward vs episode steps
        self.line_mean_reward_steps, = self.ax_steps.plot([], [], 'b-')
        self.ax_steps.set_ylabel('Average reward')
        self.ax_steps.set_xlabel('Iteration')
        self.ax_steps.xaxis.label.set_visible(True)
        self.ax_steps.set_yscale('log')
        
    def update_reward_plot(self, start_time):
        rewards = self.memory.rewards.detach().numpy()

        elapsed_time = time.time() - start_time
        self.total_time_elapsed += elapsed_time
        self.wall_clock_times.append(self.total_time_elapsed)
        self.x_data.append(self.iteration)

        self.mean_reward.append(np.mean(rewards))
        self.iteration += 1

        # Update plot for mean reward vs wall clock time
        self.line_mean_reward_time.set_data(self.wall_clock_times, self.mean_reward)
        self.ax_time.relim()
        self.ax_time.autoscale_view()

        # Update plot for mean reward vs episode steps
        self.line_mean_reward_steps.set_data(self.x_data, self.mean_reward)
        self.ax_steps.relim()
        self.ax_steps.autoscale_view()

        self.fig_reward.canvas.draw()
        self.fig_reward.canvas.flush_events()

    def save_figure(self):
        self.fig.savefig("myfigure.svg", format='svg')
    
    def save_reward_plot(self):
        self.fig_reward.tight_layout()
        self.fig_reward.savefig(f"myfigurereward.svg", format='svg')

    def unroll_environments(self):
        self.memory.clear()

        for ienv, env in enumerate(self.environments):
            env.clear_gradients()
            observation = env.get_observation()

            for istep in range(self.episode_length):
                action = self.policy(observation)
                next_obs, reward, penalty = env.step(action[0])

                self.memory.store(ienv, istep, observation, next_obs, action, reward, penalty)
                observation = next_obs

    def calculate_policy_loss(self):
        with torch.no_grad():
            target_values = self.target_critic(self.memory.next_obs[:, -1, :]).squeeze(dim=1)

        gamma_vec = self.GAMMA ** torch.arange(self.episode_length)
        discounted_rewards = torch.sum(gamma_vec * self.memory.rewards.squeeze(dim=2), dim=1)
        discounted_rewards = discounted_rewards + self.GAMMA ** self.episode_length * target_values

        policy_loss = -torch.sum(discounted_rewards) / (self.episode_length * self.num_environments)
        return policy_loss

    def calculate_critic_loss(self):
        V     = torch.zeros((self.num_environments, self.episode_length))
        V_est = torch.zeros((self.num_environments, self.episode_length))

        for t in range(self.episode_length):
            V_est[:, t] = self.calculate_estimated_value(t)
            V[:, t]     = self.critic(self.memory.observations[:, t, :].detach()).squeeze()

        # V_est = self.calculate_kth_step_return(t=0, k=self.episode_length)
        # V     = self.critic(self.memory.observations[:, 0, :].detach()).squeeze()

        critic_loss = F.mse_loss(V, V_est)
        return critic_loss

    @torch.no_grad()
    def calculate_kth_step_return(self, t, k):
        target_values = self.critic(self.memory.next_obs[:, t + k - 1, :]).squeeze(dim=1)

        gamma_vec = self.GAMMA ** torch.arange(k)
        G_t_k = torch.sum(gamma_vec * self.memory.rewards[:, t:t + k].squeeze(dim=2), dim=1)
        G_t_k = G_t_k + self.GAMMA ** k * target_values

        return G_t_k

    @torch.no_grad()
    def calculate_estimated_value(self, t):
        ks = torch.arange(1, self.episode_length - t)

        V_est = self.LAMBDA ** (self.episode_length - t - 1) * self.calculate_kth_step_return(t, self.episode_length - t)
        if len(ks):
            lambda_vec = self.LAMBDA ** ks
            G_t_ks = torch.stack([self.calculate_kth_step_return(t, k) for k in ks], dim=1)
            V_est  = V_est + (1 - self.LAMBDA) * torch.sum(lambda_vec * G_t_ks)

        return V_est

    def train_policy(self, num_episodes=10000):
        self.policy.train()
        self.critic.train()

        for episode in range(num_episodes):
            start_time = time.time()
            # Sample N short-horizon trajectories
            self.unroll_environments()

            # Update policy
            policy_loss = self.calculate_policy_loss()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update critic
            critic_loss = self.calculate_critic_loss()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic.parameters(), 1.0)
            clip_grad_value_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Soft update target critic
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.alpha * target_param.data + (1.0 - self.alpha) * param.data)

            self.update_plot(policy_loss.item(), critic_loss.item(), start_time)
            self.update_reward_plot(start_time)
            print(f"Episode: {episode}, Policy loss: {policy_loss.item():.3f}, Critic loss: {critic_loss.item():.3f}")


            # Save last model
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
            }, './models/shac_last_model.pt')

            # Check and save the best model
            current_avg_reward = np.mean(self.memory.rewards.detach().numpy())
            if current_avg_reward > self.best_avg_reward:
                self.best_avg_reward = current_avg_reward
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                }, './models/shac_best_model.pt')
            
            #self.fig.savefig("shac_last_figure.svg", format='svg')
            self.fig_reward.tight_layout()
            self.fig_reward.savefig("shac_learning_curve.svg", format='svg')
