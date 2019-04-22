"""
Unit tests for train_pg_f18.py
"""

from sklearn import preprocessing
import numpy as np
from train_pg_f18 import Agent
from mock import patch

class TestPolicyGradients(object):
    def test_normalize(self):
        with patch.object(Agent, "__init__", lambda p1, p2, p3, p4: None):
            agent = Agent(None, None, None)
            a=np.array([1, -13, 44, 100, 57, -20, 53, 53, 6, 0])
            np.testing.assert_allclose(agent.norm(a), preprocessing.scale(a))


    def test_sum_of_rewards_monte_carlo(self):
        with patch.object(Agent, "__init__", lambda p1, p2, p3, p4: None):
            agent = Agent(None, None, None)
            agent.reward_to_go = False
            agent.gamma = 0.5
            rewards = np.array([np.ones(3), np.ones(3)])
            expected = [1.75] * 6

            actual = agent.sum_of_rewards(rewards)

            assert len(expected) == len(actual)
            np.testing.assert_allclose(expected, actual)


    def test_sum_of_rewards_reward_to_go(self):
        with patch.object(Agent, "__init__", lambda p1, p2, p3, p4: None):
            agent = Agent(None, None, None)
            agent.reward_to_go = True
            agent.gamma = 0.5
            rewards = np.array([np.ones(3), np.ones(3)])
            expected = [1.75, 1.5, 1, 1.75, 1.5, 1]

            actual = agent.sum_of_rewards(rewards)

            assert len(expected) == len(actual)
            np.testing.assert_allclose(expected, actual)
