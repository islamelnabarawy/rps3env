"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import sys
import unittest

import gym
from gym import Space

# noinspection PyUnresolvedReferences
import rps3env
from rps3env.envs import RPS3GameEnv

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

EMPTY_BOARD = """
            ..
         .. ..  ..
      ..        .. ..
   .. ..              ..
.. ..         ..      .. ..
   ..                    .. ..
      .. ..              ..
         ..        .. ..
            .. ..  ..
               ..
"""

EMPTY_OBSERVATION = ['0'] * 28

INIT_BOARD = """
            OP
         OR ..  OS
      OP        .. OR
   OS ..              OS
OP ..         ..      .. OR
   PS                    .. PR
      PP ..              PP
         PR        .. PS
            PS ..  PR
               PP
"""

OBS_AFTER_BOARD_INIT = ['PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', 'PS'] + \
                       ['OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU'] + \
                       ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

OBS_AFTER_LEGAL_MOVE = ['0', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', 'PS'] + \
                       ['OU', 'OU', 'OU', 'OU', 'OU', 'OU', '0', 'OU', 'OU'] + \
                       ['PR', '0', '0', '0', '0', '0', '0', 'OU', '0', '0']
OBS_AFTER_CHALLENGE_WIN = ['PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', '0'] + \
                          ['PS', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', '0', 'OU'] + \
                          ['0', '0', '0', '0', '0', '0', '0', '0', 'OU', '0']
OBS_AFTER_CHALLENGE_TIE = ['PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', 'PS'] + \
                          ['OU', 'OU', 'OU', 'OU', 'OU', 'OU', '0', 'OU', 'OR'] + \
                          ['0', '0', '0', '0', '0', '0', '0', 'OU', '0', '0']
OBS_AFTER_CHALLENGE_LOSS = ['PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', '0'] + \
                           ['OR', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OR'] + \
                           ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
OBS_AFTER_FULL_GAME = ['PR', 'PP', 'PS', 'PR', 'PP', 'PS', '0', 'PP', '0'] + \
                      ['0', '0', '0', '0', 'PS', '0', '0', 'OU', '0'] + \
                      ['OU', '0', '0', '0', '0', 'OS', 'OU', 'OU', 'OR', 'PR']


class RPS3GameEnvTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.env = gym.make('RPS3Game-v0')  # type: RPS3GameEnv

    def tearDown(self):
        super().tearDown()
        self.env.close()

    def init_board(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(['R', 'P', 'S'] * 3)
        return obs, reward, done, info

    def step_assert(self, obs_actual, reward_actual, done_actual, info_actual,
                    obs_expected, reward_expected=None, done_expected=False, info_expected=None):
        if reward_expected is None:
            reward_expected = [0, 0]
        if info_expected is None:
            info_expected = {'turn': 0}
        self.assertEqual(obs_expected, obs_actual, msg='Arrays are not equal.')
        self.assertEqual(reward_expected, reward_actual)
        self.assertEqual(done_expected, done_actual)
        self.assertEqual(info_expected, info_actual)

    def test_initializable(self):
        self.assertIsNotNone(self.env, msg='gym.make() returned None.')

    def test_action_space_pre_reset(self):
        self.assertRaises(ValueError, lambda: self.env.action_space)

    def test_action_space_pre_init(self):
        self.env.reset()
        self.assertIsInstance(self.env.action_space, Space)
        self.assertEqual(9, self.env.action_space.shape)
        self.assertEqual(3, max(self.env.action_space.high))
        self.assertEqual(3, min(self.env.action_space.high))
        self.assertEqual(1, max(self.env.action_space.low))
        self.assertEqual(1, min(self.env.action_space.low))

    def test_action_space_post_init(self):
        self.env.reset()
        self.env.step(['R', 'P', 'S'] * 3)
        self.assertIsInstance(self.env.action_space, Space)
        self.assertEqual(2, self.env.action_space.shape)
        self.assertEqual(27, max(self.env.action_space.high))
        self.assertEqual(27, min(self.env.action_space.high))
        self.assertEqual(0, max(self.env.action_space.low))
        self.assertEqual(0, min(self.env.action_space.low))

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space, Space)
        self.assertEqual(2, len(self.env.observation_space.spaces))
        self.assertEqual(28, self.env.observation_space.spaces[0].shape)
        self.assertEqual(4, max(self.env.observation_space.spaces[0].high))
        self.assertEqual(4, min(self.env.observation_space.spaces[0].high))
        self.assertEqual(0, max(self.env.observation_space.spaces[0].low))
        self.assertEqual(0, min(self.env.observation_space.spaces[0].low))
        self.assertEqual(28, self.env.observation_space.spaces[1].n)

    def test_reward_range(self):
        self.assertEqual(-100, self.env.reward_range[0])
        self.assertEqual(100, self.env.reward_range[1])

    def test_reset(self):
        actual = self.env.reset()
        expected = EMPTY_OBSERVATION
        self.assertEqual(expected, actual, msg='Arrays are not equal.')

    def test_render_empty_board(self):
        self.env.reset()
        actual = self.env.render(mode='ansi')
        expected = EMPTY_BOARD
        self.assertEqual(expected, actual)

    def test_set_board(self):
        self.env.seed()
        obs, reward, done, info = self.init_board()
        self.step_assert(obs, reward, done, info, OBS_AFTER_BOARD_INIT)

    def test_render_set_board(self):
        self.env.seed(0)
        self.init_board()
        actual = self.env.render(mode='ansi')
        expected = INIT_BOARD
        self.assertEqual(expected, actual, msg='Arrays are not equal.')

    def test_malformed_move(self):
        self.env.seed(0)
        self.init_board()

        def make_empty_move():
            self.env.step(())

        self.assertRaises(AssertionError, make_empty_move)

        def make_bad_move():
            self.env.step(('',))

        self.assertRaises(AssertionError, make_bad_move)

    def test_illegal_moves(self):
        self.env.seed(0)
        self.init_board()

        def make_illegal_move_1():
            self.env.step(('O0', 'O1'))

        self.assertRaises(AssertionError, make_illegal_move_1)

        def make_illegal_move_2():
            self.env.step(('O0', 'I1'))

        self.assertRaises(AssertionError, make_illegal_move_2)

        def make_illegal_move_3():
            self.env.step(('O0', 'C0'))

        self.assertRaises(AssertionError, make_illegal_move_3)

    def test_legal_move(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step(('O0', 'I0'))
        self.step_assert(obs, reward, done, info, OBS_AFTER_LEGAL_MOVE, info_expected={'turn': 1})

    def test_challenge_tie(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step(('O0', 'O17'))
        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_TIE, info_expected={'turn': 1})

    def test_challenge_win(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step(('O8', 'O9'))
        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_WIN,
                         reward_expected=[1, 0], info_expected={'turn': 1})

    def test_challenge_loss(self):
        self.env.seed(2)
        self.init_board()
        obs, reward, done, info = self.env.step(('O8', 'O9'))
        self.step_assert(obs, reward, done, info, OBS_AFTER_CHALLENGE_LOSS,
                         reward_expected=[-1, 0], info_expected={'turn': 1})

    def test_full_game(self):
        self.env.seed(0)
        self.init_board()
        moves = [
            ('O8', 'O9'), ('O9', 'I4'), ('I4', 'I5'), ('I5', 'O11'), ('O11', 'I5'),
            ('I5', 'O11'), ('O11', 'O12'), ('O12', 'O13'), ('O6', 'I3')
        ]
        for move in moves:
            self.env.step(move)
        obs, reward, done, info = self.env.step(('I3', 'C0'))
        self.step_assert(obs, reward, done, info, OBS_AFTER_FULL_GAME,
                         reward_expected=[100, 0], done_expected=True, info_expected={'turn': 10})


if __name__ == '__main__':
    unittest.main()
