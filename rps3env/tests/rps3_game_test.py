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
import unittest
import gym
import numpy as np

# noinspection PyUnresolvedReferences
import rps3env

__author__ = 'Islam Elnabarawy'


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
            OU
         OU ..  OU
      OU        .. OU
   OU ..              OU
OU ..         ..      .. OU
   PS                    .. PR
      PP ..              PP
         PR        .. PS
            PS ..  PR
               PP
"""


class RPS3GameEnvTest(unittest.TestCase):
    def test_initializable(self):
        env = gym.make('RPS3Game-v0')
        self.assertIsNotNone(env, msg='gym.make() returned None.')

    def test_reset(self):
        env = gym.make('RPS3Game-v0')
        actual = env.reset()
        expected = np.array(['0']*28)
        self.assertEqual(type(expected), type(actual), msg='Types are not equal.')
        self.assertTrue(np.array_equal(expected, actual), msg='Arrays are not equal.')

    def test_render_empty_board(self):
        env = gym.make('RPS3Game-v0')
        env.reset()
        actual = env.render(mode='ansi')
        expected = EMPTY_BOARD
        self.assertEqual(expected, actual)

    def test_set_board(self):
        env = gym.make('RPS3Game-v0')
        env.reset()
        env.step(['R', 'P', 'S'] * 3)
        actual = env.render(mode='ansi')
        expected = INIT_BOARD
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
