import pandas as pd
import torch
from dataclasses import dataclass
from source.cstr import CSTR, CSTRParameters
from torch.nn import functional as F


SEC_PER_HR = 60.0 * 60.0  # 60min.


@dataclass
class EnvironmentParameters:
    delta_t                    = SEC_PER_HR
    storage_size               = 24 
    nominal_production_rate    = 1.0 / SEC_PER_HR
    nominal_coolant_flow_rate  = 390.0 / SEC_PER_HR
    price_smoothing            = 0.9

    x_SS = {'c'  : 0.1367,
            'roh': 1.0 / SEC_PER_HR,  # production rate
            'T'  : 0.7293,
            'Fc' : 390.0 / SEC_PER_HR}  # coolant flow rate

    cstr_params = CSTRParameters(integration_method='rk4')
    price_prediction_horizon = 12


class Environment(torch.nn.Module):
    def __init__(self, parameters: EnvironmentParameters, prices, initial_step, max_step):
        super().__init__()

        self.params = parameters
        self.CSTR   = CSTR(self.params.cstr_params)

        self.prices        = prices
        self.average_price = 0.0
        self.max_step      = max_step
        self.current_step  = initial_step

        self.min_state   = torch.tensor([0.9 * self.params.x_SS['c'], 0.6])
        self.max_state   = torch.tensor([1.1 * self.params.x_SS['c'], 0.8])
        self.min_storage = 0.0
        self.max_storage = self.params.storage_size

        self.state_mid     = 0.5 * (self.max_state + self.min_state)
        self.storage_mid   = 0.5 * (self.max_storage + self.min_storage)
        self.state_range   = self.max_state - self.min_state
        self.storage_range = self.max_storage - self.min_storage

        self.reset()

    def step(self, action):
        # update average price
        self.average_price = self.params.price_smoothing * self.average_price + (1 - self.params.price_smoothing) * self.get_price_prediction().mean()

        # solve the initial value problem given by the current state and the chosen action
        self.state   = self.CSTR.forward(self.state, action, self.params.delta_t)
        self.storage = self.storage + (action[0] - self.params.nominal_production_rate) / self.params.nominal_production_rate
        observation  = self.get_observation()

        # calculate cost savings (assume coolant most costly)
        nominal_cost = self.params.nominal_coolant_flow_rate * self.prices[self.current_step] * self.params.delta_t
        cost         = action[1] * self.prices[self.current_step] * self.params.delta_t

        # calculates the total penalty based on how much self.state and self.storage deviate from their allowed ranges
        # as a quadratic function =-1 when in the middle of the range and =0 at the bounds.
        state_penalty   = (self.state - self.state_mid)**2 / (self.state_mid - self.min_state)**2 - 1
        storage_penalty = (self.storage - 1 - self.storage_mid)**2 / (self.storage_mid - self.min_storage)**2 - 1

        penalty = F.relu(torch.concat([state_penalty, storage_penalty])).mean()
        reward  = torch.tanh(1e-2 * (nominal_cost - cost) / (self.params.nominal_coolant_flow_rate * self.params.delta_t)) - penalty

        # after completing one full cycle, it goes back to the starting position.
        self.current_step = (self.current_step + 1) % self.max_step

        return observation, reward, penalty

    # removes gradient information between simulation runs to prevent from multiple backward passes through the same graph.
    # I.e., after each call of "policy_loss.backward()", the gradient information needs to be reset, but we don't want to reset the entire environment.
    def clear_gradients(self):
        with torch.no_grad():
            current_state   = self.state.clone()
            current_storage = self.storage.clone()
            self.state      = current_state
            self.storage    = current_storage

    # define initial
    def reset(self):
        self.state   = torch.tensor([self.params.x_SS["c"], self.params.x_SS["T"]], requires_grad=True)
        self.storage = torch.tensor([1.0], requires_grad=True)  # [hours of steady state production]

        return self.get_observation()

    def get_price_prediction(self) -> torch.Tensor:
        return self.prices[self.current_step:self.current_step + self.params.price_prediction_horizon]

    def get_observation(self):
        return torch.cat((self.state, self.storage, self.get_price_prediction() - self.average_price), dim=0)

    @staticmethod
    def load_prices(price_prediction_horizon: int):
        prices = pd.read_excel('data/consecutive_prices.xlsx')
        prices = torch.from_numpy(prices.loc[:, 'AT_price_day_ahead'].to_numpy()).type(torch.float32)

        max_step = len(prices)
        prices = torch.concat([prices, prices[:price_prediction_horizon]])

        return prices, max_step
