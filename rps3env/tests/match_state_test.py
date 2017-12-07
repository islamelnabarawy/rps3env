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
import copy
import unittest

from rps3env.opponents.match_state import MatchState
from rps3env.tests.base_opponent_test import TestBaseOpponent

__author__ = 'Islam Elnabarawy'


class TestMatchStatePieceProbabilities(unittest.TestCase):

    def test_getDefaultBoardProbabilities(self):
        state = MatchState(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        probabilities = state.get_opponent_piece_probabilities()
        self.assertEqual([1.0 / 3] * 3, probabilities)

    def test_getEmptyBoardProbabilities(self):
        state = MatchState()
        probabilities = state.get_opponent_piece_probabilities()
        self.assertEqual([0.0 / 3] * 3, probabilities)


class TestMatchStatePieceCounts(unittest.TestCase):

    def test_getDefaultBoardCounts(self):
        state = MatchState(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        self.assertEqual([0] * 3 + [9], state._opponent_counts)

    def test_getEmptyBoardCounts(self):
        state = MatchState()
        self.assertEqual([0] * 4, state._opponent_counts)


class TestMatchStateClone(unittest.TestCase):

    def test_cloneDifferent(self):
        state1 = MatchState()
        state2 = state1.clone()
        self.assertIsNot(state1, state2)

    def test_cloneDeepCopy(self):
        state1 = MatchState()
        state2 = state1.clone()
        self.assertIsNot(state1._board, state2._board)
        self.assertIsNot(state1._captures, state2._captures)
        self.assertIsNot(state1._opponent_counts, state2._opponent_counts)


class TestMatchStateMatchOver(unittest.TestCase):

    def test_notOver(self):
        state = MatchState(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        prob_match_over, winner = state.is_match_over()
        self.assertEqual(0.0, prob_match_over)

    def test_playerWinCenter(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][9:12] = ['0'] * 3
        board['O'][2] = '0'
        board['C'][0] = 'PS'
        state = MatchState(board, [3, 0, 0], [3, 0, 0, 6], [3, 3, 3])
        prob_match_over, winner = state.is_match_over()
        self.assertEqual(1.0, prob_match_over)
        self.assertEqual('P', winner)

    def test_opponentWinCenterKnown(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][0:9:3] = ['0'] * 3
        board['O'][9] = '0'
        board['C'][0] = 'OS'
        state = MatchState(board, [0, 0, 0], [0, 0, 1, 8], [0, 3, 3])
        prob_match_over, winner = state.is_match_over()
        self.assertEqual(1.0, prob_match_over)
        self.assertEqual('O', winner)

    def test_opponentWinCenterOneUnknown(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][0:9] = ['PR'] * 3 + ['PP'] * 3 + ['PS'] * 3
        state = MatchState(board)
        state.apply_move(('O', 17), ('O', 0), 'W', 'P')  # opponent
        state.apply_move(('O', 1), ('O', 0), 'L', 'P')  # player
        state.apply_move(('O', 0), ('O', 1), 'M')  # opponent
        state.apply_move(('O', 2), ('O', 1), 'L', 'P')  # player
        state.apply_move(('O', 9), ('I', 4), 'M')  # opponent
        state.apply_move(('O', 3), ('O', 2), 'M')  # player
        state.apply_move(('I', 4), ('C', 0), 'M')  # opponent
        prob_match_over, winner = state.is_match_over()
        prob_pieces = state.get_opponent_piece_probabilities()
        self.assertEqual(prob_pieces[2], prob_match_over)
        self.assertEqual('O', winner)

    def test_opponentWinCenterTwoUnknowns(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][0:9] = ['PR'] * 3 + ['PP'] * 3 + ['PS'] * 3
        state = MatchState(board)
        state.apply_move(('O', 17), ('O', 0), 'W', 'P')  # opponent
        state.apply_move(('O', 1), ('O', 0), 'L', 'P')  # player
        state.apply_move(('O', 0), ('O', 1), 'M')  # opponent
        state.apply_move(('O', 2), ('O', 1), 'L', 'P')  # player
        state.apply_move(('O', 9), ('O', 8), 'W', 'R')  # opponent
        state.apply_move(('O', 7), ('O', 8), 'L', 'R')  # player
        state.apply_move(('O', 8), ('O', 7), 'M')  # opponent
        state.apply_move(('O', 6), ('O', 7), 'L', 'R')  # player
        state.apply_move(('O', 13), ('I', 6), 'M')  # opponent
        state.apply_move(('O', 3), ('O', 2), 'M')  # player
        state.apply_move(('I', 6), ('C', 0), 'M')  # opponent
        prob_match_over, winner = state.is_match_over()
        prob_pieces = state.get_opponent_piece_probabilities()
        self.assertEqual(prob_pieces[1] + prob_pieces[2], prob_match_over)
        self.assertEqual('O', winner)

    def test_playerWinNoPieces(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][9:] = ['0'] * 9
        state = MatchState(board, [3, 3, 3], [3, 3, 3, 0], [3, 3, 3])
        prob_match_over, winner = state.is_match_over()
        self.assertEqual(1.0, prob_match_over)
        self.assertEqual('P', winner)

    def test_opponentWinNoPieces(self):
        board = copy.deepcopy(TestBaseOpponent.DEFAULT_BLUE_BOARD)
        board['O'][:9] = ['0'] * 9
        state = MatchState(board, [0, 0, 0], [0, 0, 0, 9], [0, 0, 0])
        prob_match_over, winner = state.is_match_over()
        self.assertEqual(1.0, prob_match_over)
        self.assertEqual('O', winner)
