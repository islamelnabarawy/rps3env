import unittest
import gym

# noinspection PyUnresolvedReferences
import rps3env


class RPS3GameEnvTest(unittest.TestCase):
    def test_initializable(self):
        env = gym.make('RPS3Game-v0')
        self.assertIsNotNone(env)


if __name__ == '__main__':
    unittest.main()
