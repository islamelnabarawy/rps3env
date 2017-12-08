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
from gym import Space, spaces

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

OBS_BEFORE_BOARD_INIT = {
    'occupied': [False] * 28,
    'player_owned': [False] * 28,
    'piece_type': [-1] * 28
}

OBS_AFTER_BOARD_INIT = {
    'occupied': [True] * 18 + [False] * 10,
    'player_owned': [True] * 9 + [False] * 19,
    'piece_type': [1, 2, 3] * 3 + [0] * 9 + [-1] * 10
}

OBS_AFTER_LEGAL_MOVE = {
    'occupied': [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True,
                 True, True, False, False, False, False, False, False, True, False, False],
    'player_owned': [False, True, True, True, True, True, True, True, True, False, False, False, False, False, False,
                     False, False, False, True, False, False, False, False, False, False, False, False, False],
    'piece_type': [-1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1]
}

OBS_AFTER_CHALLENGE_WIN = {
    'occupied': [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False,
                 True, False, False, False, False, False, False, False, False, True, False],
    'player_owned': [True, True, True, True, True, True, True, True, False, True, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, -1, 3, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1]
}

OBS_AFTER_CHALLENGE_TIE = {
    'occupied': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True,
                 True, False, False, False, False, False, False, False, True, False, False],
    'player_owned': [True, True, True, True, True, True, True, True, True, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0, 0, 0, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1]
}

OBS_AFTER_CHALLENGE_LOSS = {
    'occupied': [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True,
                 True, False, False, False, False, False, False, False, False, False, False],
    'player_owned': [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, False],
    'piece_type': [1, 2, 3, 1, 2, 3, 1, 2, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
}

OBS_AFTER_FULL_GAME = {
    'occupied': [True, True, True, True, True, True, False, True, False, False, False, False, False, True, False, False,
                 True, False, True, False, False, False, False, True, True, True, True, True],
    'player_owned': [True, True, True, True, True, True, False, True, False, False, False, False, False, True, False,
                     False, False, False, False, False, False, False, False, False, False, False, False, True],
    'piece_type': [1, 2, 3, 1, 2, 3, -1, 2, -1, -1, -1, -1, -1, 3, -1, -1, 0, -1, 0, -1, -1, -1, -1, 3, 0, 0, 1, 1]
}


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
        self.assertIsInstance(self.env.observation_space, spaces.Dict)
        self.assertEqual(3, len(self.env.observation_space.spaces))
        self.assertEqual(28, self.env.observation_space.spaces['occupied'].n)
        self.assertEqual(28, self.env.observation_space.spaces['player_owned'].n)
        self.assertEqual(28, self.env.observation_space.spaces['piece_type'].shape)
        self.assertEqual(3, max(self.env.observation_space.spaces['piece_type'].high))
        self.assertEqual(3, min(self.env.observation_space.spaces['piece_type'].high))
        self.assertEqual(-1, max(self.env.observation_space.spaces['piece_type'].low))
        self.assertEqual(-1, min(self.env.observation_space.spaces['piece_type'].low))

    def test_reward_range(self):
        self.assertEqual(-100, self.env.reward_range[0])
        self.assertEqual(100, self.env.reward_range[1])

    def test_reset(self):
        actual = self.env.reset()
        expected = OBS_BEFORE_BOARD_INIT
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
