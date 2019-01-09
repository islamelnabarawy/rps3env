"""
   Copyright 2019 Islam Elnabarawy

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
from rps3env.opponents import RandomOpponent
from rps3env.tests.base_opponent_test import TestBaseOpponent

__author__ = 'Islam Elnabarawy'


class TestRandomOpponent(TestBaseOpponent):

    def test_randomLegalBlueMove(self):
        opponent = RandomOpponent()
        opponent.init_board_layout(0)
        move = opponent.get_next_move()
        self.assertIn(move, self.POSSIBLE_BLUE_MOVES)

    def test_randomLegalGreenMove(self):
        opponent = RandomOpponent()
        opponent.init_board_layout(1)
        move = opponent.get_next_move()
        self.assertIn(move, self.POSSIBLE_GREEN_MOVES)

    def test_randomLegalBlueLayout(self):
        opponent = RandomOpponent()
        opponent.init_board_layout(0)
        layout = opponent.board['O'][:9]
        self.assertEqual(layout.count('PR'), 3)
        self.assertEqual(layout.count('PP'), 3)
        self.assertEqual(layout.count('PS'), 3)

    def test_randomLegalGreenLayout(self):
        opponent = RandomOpponent()
        opponent.init_board_layout(1)
        layout = opponent.board['O'][9:]
        self.assertEqual(layout.count('PR'), 3)
        self.assertEqual(layout.count('PP'), 3)
        self.assertEqual(layout.count('PS'), 3)

    def test_randomLegalPlayerHand(self):
        opponent = RandomOpponent()
        opponent.init_board_layout(0)
        hand = opponent.get_player_hand()
        self.assertIn(hand, ['R', 'P', 'S'])
