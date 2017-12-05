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

# noinspection PyUnresolvedReferences
import rps3env

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

INIT_OBSERVATION = ['PR', 'PP', 'PS'] * 3 + ['OU'] * 9 + ['0'] * 10

OBS_AFTER_LEGAL_MOVE = ['0', 'PP', 'PS'] + ['PR', 'PP', 'PS'] * 2 + ['OU'] * 9 + ['PR'] + ['0'] * 9
OBS_AFTER_CHALLENGE_WIN = ['PR', 'PP', 'PS'] * 2 + ['PR', 'PP', '0', 'PS'] + ['OU'] * 8 + ['0'] * 10
OBS_AFTER_CHALLENGE_LOSS = ['PR', 'PP', 'PS'] * 2 + ['PR', 'PP', '0'] + ['OU'] * 9 + ['0'] * 10


class RPS3GameEnvTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.env = gym.make('RPS3Game-v0')

    def tearDown(self):
        super().tearDown()
        self.env.close()

    def init_board(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(['R', 'P', 'S'] * 3)
        return done, info, obs, reward

    def test_initializable(self):
        self.assertIsNotNone(self.env, msg='gym.make() returned None.')

    def test_reset(self):
        actual = self.env.reset()
        expected = EMPTY_OBSERVATION
        self.assertEqual(type(expected), type(actual), msg='Types are not equal.')
        self.assertEqual(expected, actual, msg='Arrays are not equal.')

    def test_render_empty_board(self):
        self.env.reset()
        actual = self.env.render(mode='ansi')
        expected = EMPTY_BOARD
        self.assertEqual(expected, actual)

    def test_set_board(self):
        self.env.seed()
        done, info, obs, reward = self.init_board()
        expected = INIT_OBSERVATION
        self.assertEqual(expected, obs, msg='Arrays are not equal.')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual({'turn': 0}, info)

    def test_render_set_board(self):
        self.env.seed(0)
        self.init_board()
        actual = self.env.render(mode='ansi')
        expected = INIT_BOARD
        self.assertEqual(expected, actual, msg='Arrays are not equal.')

    def test_malformed_move(self):
        self.init_board()

        def make_empty_move():
            self.env.step(())

        self.assertRaises(AssertionError, make_empty_move)

        def make_bad_move():
            self.env.step(('',))

        self.assertRaises(AssertionError, make_bad_move)

    def test_illegal_moves(self):
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
        self.init_board()
        obs, reward, done, info = self.env.step(('O0', 'I0'))
        expected = OBS_AFTER_LEGAL_MOVE
        self.assertEqual(type(expected), type(obs), msg='Types are not equal.')
        self.assertEqual(expected, obs, msg='Arrays are not equal.')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual({'turn': 1}, info)

    def test_challenge_tie(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step(('O0', 'O17'))
        expected = INIT_OBSERVATION
        self.assertEqual(type(expected), type(obs), msg='Types are not equal.')
        self.assertEqual(expected, obs, msg='Arrays are not equal.')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual({'turn': 1}, info)

    def test_challenge_win(self):
        self.env.seed(0)
        self.init_board()
        obs, reward, done, info = self.env.step(('O8', 'O9'))
        expected = OBS_AFTER_CHALLENGE_WIN
        self.assertEqual(type(expected), type(obs), msg='Types are not equal.')
        self.assertEqual(expected, obs, msg='Arrays are not equal.')
        self.assertEqual(1, reward)
        self.assertEqual(False, done)
        self.assertEqual({'turn': 1}, info)

    def test_challenge_lose(self):
        self.env.seed(2)
        self.init_board()
        obs, reward, done, info = self.env.step(('O8', 'O9'))
        expected = OBS_AFTER_CHALLENGE_LOSS
        self.assertEqual(type(expected), type(obs), msg='Types are not equal.')
        self.assertEqual(expected, obs, msg='Arrays are not equal.')
        self.assertEqual(-1, reward)
        self.assertEqual(False, done)
        self.assertEqual({'turn': 1}, info)


if __name__ == '__main__':
    unittest.main()
