import torch

# TODO: try RNN e.g. GRU or with buffer to recall historical information?
# TODO: try expanding layers?
class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        sec_per_hour = 60.0 * 60.0

        self.state_input_layer   = torch.nn.Linear(2, 15)
        self.storage_input_layer = torch.nn.Linear(1, 15)
        self.prices_input_layer  = torch.nn.Linear(12, 20)

        self.input_layer  = torch.nn.Linear(50, 50)  # len(observation) = 15
        self.hidden_layer = torch.nn.Linear(50, 50)
        self.output_layer = torch.nn.Linear(50, 2)  # len(action) = 2

        self.max_action = torch.tensor([1.2 / sec_per_hour, 700.0 / sec_per_hour])
        self.min_action = torch.tensor([0.8 / sec_per_hour, 0.0 / sec_per_hour])
        self.max_state  = torch.tensor([1.1 * 0.1367, 0.8])  # c_upper, T_upper
        self.min_state  = torch.tensor([0.9 * 0.1367, 0.6])  # c_lower, T_lower

    # use consistent activation functions
    def forward(self, observation) -> torch.Tensor:
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        # extract different parts of the current environment state
        state   = self.scale_state(observation[:, :2])
        storage = observation[:, 2].unsqueeze(1)
        prices  = observation[:, 3:]

        # pass through network
        state   = torch.tanh(self.state_input_layer(state))
        storage = torch.tanh(self.storage_input_layer(storage))
        prices  = torch.tanh(self.prices_input_layer(prices))

        x = torch.concat((state, storage, prices), dim=1)
        x = torch.tanh(self.input_layer(x))
        x = torch.tanh(self.hidden_layer(x))

        # TODO: try sample from normal dist. 
        # TODO: try stochastic action
        action = torch.sigmoid(self.output_layer(x))
        action = self.scale_action(action)
        return action

    def scale_action(self, action):
        action = action * (self.max_action - self.min_action) + self.min_action
        return action

    # any values of state between its extremes is scaled proportionally between 0 and 1.
    def scale_state(self, state):
        state = (state - self.min_state) / (self.max_state - self.min_state)
        return state
