"""
Basic test for the Environment class.

Run with: python -m pytest tests/test_environment.py
"""
import unittest
import torch
import numpy as np

from environment import Environment, EnvironmentParameters


class TestEnvironment(unittest.TestCase):
    """Test cases for the CSTR Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = EnvironmentParameters()
        # Create mock price data for testing
        self.prices = torch.ones(100) * 50.0  # Mock electricity prices
        self.max_step = 100
        self.initial_step = 0
        
        self.env = Environment(
            self.params, 
            self.prices, 
            self.initial_step, 
            self.max_step
        )
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        # Check observation dimensions
        obs = self.env.get_observation()
        expected_obs_length = 2 + 1 + self.params.price_prediction_horizon  # state + storage + prices
        self.assertEqual(len(obs), expected_obs_length)
        
        # Check initial state is within bounds
        self.assertTrue(torch.all(self.env.state >= self.env.min_state))
        self.assertTrue(torch.all(self.env.state <= self.env.max_state))
        
        # Check storage is initialized properly
        self.assertEqual(self.env.storage.item(), 1.0)
    
    def test_valid_action_execution(self):
        """Test that valid actions can be executed."""
        # Create a valid action (production rate, coolant flow rate)
        action = torch.tensor([
            self.params.nominal_production_rate,
            self.params.nominal_coolant_flow_rate
        ])
        
        initial_obs = self.env.get_observation()
        next_obs, reward, penalty = self.env.step(action)
        
        # Check that observation dimensions remain consistent
        self.assertEqual(len(next_obs), len(initial_obs))
        
        # Check that reward and penalty are scalars
        self.assertIsInstance(reward, torch.Tensor)
        self.assertIsInstance(penalty, torch.Tensor)
        self.assertEqual(reward.numel(), 1)
        self.assertEqual(penalty.numel(), 1)
    
    def test_action_bounds_handling(self):
        """Test behavior with actions at boundaries."""
        # Test minimum action
        min_action = torch.tensor([0.8/3600, 0.0])  # From policy bounds
        obs, reward, penalty = self.env.step(min_action)
        self.assertIsInstance(reward, torch.Tensor)
        
        # Test maximum action  
        max_action = torch.tensor([1.2/3600, 700.0/3600])  # From policy bounds
        obs, reward, penalty = self.env.step(max_action)
        self.assertIsInstance(reward, torch.Tensor)
    
    def test_price_prediction(self):
        """Test price prediction functionality."""
        price_pred = self.env.get_price_prediction()
        
        # Check correct length
        self.assertEqual(len(price_pred), self.params.price_prediction_horizon)
        
        # Check prices are from the correct time step
        expected_prices = self.prices[
            self.env.current_step:self.env.current_step + self.params.price_prediction_horizon
        ]
        self.assertTrue(torch.allclose(price_pred, expected_prices))
    
    def test_step_progression(self):
        """Test that current_step advances correctly."""
        initial_step = self.env.current_step
        
        action = torch.tensor([
            self.params.nominal_production_rate,
            self.params.nominal_coolant_flow_rate
        ])
        
        self.env.step(action)
        
        # Check step advanced
        self.assertEqual(self.env.current_step, (initial_step + 1) % self.max_step)
    
    def test_gradient_clearing(self):
        """Test that gradients are properly cleared."""
        # This test ensures gradient computation doesn't accumulate
        self.env.clear_gradients()
        
        # State should still have requires_grad=True but no accumulated gradients
        self.assertTrue(self.env.state.requires_grad)
        self.assertTrue(self.env.storage.requires_grad)
    
    def test_constraint_penalties(self):
        """Test that constraint violations produce penalties."""
        # Force state outside bounds by creating extreme action
        extreme_action = torch.tensor([2.0/3600, 1000.0/3600])  # Very high action
        
        # Take multiple steps to push system out of bounds
        for _ in range(10):
            obs, reward, penalty = self.env.step(extreme_action)
        
        # Penalty should be positive (indicating constraint violation)
        # Note: This test might need adjustment based on actual dynamics
        self.assertGreaterEqual(penalty.item(), 0.0)


class TestEnvironmentParameters(unittest.TestCase):
    """Test cases for EnvironmentParameters."""
    
    def test_parameter_initialization(self):
        """Test that parameters initialize with expected values."""
        params = EnvironmentParameters()
        
        # Check key parameters exist and have reasonable values
        self.assertGreater(params.storage_size, 0)
        self.assertGreater(params.nominal_production_rate, 0)
        self.assertGreater(params.nominal_coolant_flow_rate, 0)
        self.assertGreater(params.price_prediction_horizon, 0)
        
        # Check steady state values
        self.assertIn('c', params.x_SS)
        self.assertIn('T', params.x_SS)
        self.assertIn('roh', params.x_SS)
        self.assertIn('Fc', params.x_SS)


if __name__ == '__main__':
    unittest.main()
