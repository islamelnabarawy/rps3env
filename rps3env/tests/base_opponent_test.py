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

from rps3env.opponents import BaseOpponent

__author__ = 'Islam Elnabarawy'


class ConcreteBaseOpponent(BaseOpponent):
    def _get_board_layout(self):
        return super(ConcreteBaseOpponent, self)._get_board_layout()

    def get_player_hand(self):
        return super(ConcreteBaseOpponent, self).get_player_hand()

    def get_next_move(self):
        return super(ConcreteBaseOpponent, self).get_next_move()


class TestBaseOpponent(unittest.TestCase):
    DEFAULT_BLUE_BOARD = {
        'O': [
            'PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', 'PS',
            'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU'
        ],
        'I': ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
        'C': ['0']
    }
    DEFAULT_GREEN_BOARD = {
        'O': [
            'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU', 'OU',
            'PR', 'PP', 'PS', 'PR', 'PP', 'PS', 'PR', 'PP', 'PS'
        ],
        'I': ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
        'C': ['0']
    }
    POSSIBLE_BLUE_MOVES = [
        'O0:O17', 'O0:I0', 'O1:I0', 'O2:I1', 'O3:I1', 'O4:I2', 'O5:I2',
        'O6:I3', 'O7:I3', 'O8:O9', 'O8:I4'
    ]
    POSSIBLE_GREEN_MOVES = [
        'O9:O8', 'O9:I4', 'O10:I5', 'O11:I5', 'O12:I6', 'O13:I6', 'O14:I7',
        'O15:I7', 'O16:I8', 'O17:O0', 'O17:I8'
    ]


class TestBaseOpponentEmptyBoard(TestBaseOpponent):

    def test_emptyBoardPosition(self):
        opponent = ConcreteBaseOpponent()
        move_list = opponent.get_move_list('O', 4)
        self.assertEqual([], move_list)

    def test_emptyBoardNoMoves(self):
        opponent = ConcreteBaseOpponent()
        move = opponent.get_next_move()
        self.assertIsNone(move)


class TestMinMaxOpponentBoardHash(TestBaseOpponent):
    DEFAULT_BLUE_BOARD_HASH = 'PRPPPSPRPPPSPRPPPSOUOUOUOUOUOUOUOUOU0000000000-000'
    DEFAULT_GREEN_BOARD_HASH = 'OUOUOUOUOUOUOUOUOUPRPPPSPRPPPSPRPPPS0000000000-000'

    def test_getDefaultBlueBoardHash(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        self.assertEqual(self.DEFAULT_BLUE_BOARD_HASH, opponent.get_board_hash())

    def test_getDefaultGreenBoardHash(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(1)
        self.assertEqual(self.DEFAULT_GREEN_BOARD_HASH, opponent.get_board_hash())


@unittest.skip
class TestBaseOpponentPrintBoard(TestBaseOpponent):

    def test_printEmptyBoard(self):
        opponent = ConcreteBaseOpponent()
        opponent.print_board()

    def test_printDefaultBlueBoard(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        opponent.print_board()

    def test_printDefaultGreenBoard(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(1)
        opponent.print_board()


class TestBaseOpponentDefaultBoard(TestBaseOpponent):

    def test_allLegalPlayerMoves(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        possible_moves = opponent.get_possible_moves('P')
        self.assertEqual(self.POSSIBLE_BLUE_MOVES, possible_moves)

    def test_allLegalOpponentMoves(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        possible_moves = opponent.get_possible_moves('O')
        self.assertEqual(self.POSSIBLE_GREEN_MOVES, possible_moves)

    def test_defaultBoardBlueSide(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        self.assertEqual(self.DEFAULT_BLUE_BOARD, opponent.board)
        self.assertEqual([0] * 3 + [9], opponent.counts)

    def test_defaultBoardGreenSide(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(1)
        self.assertEqual(self.DEFAULT_GREEN_BOARD, opponent.board)
        self.assertEqual([0] * 3 + [9], opponent.counts)


class TestBaseOpponentPlayerHand(TestBaseOpponent):

    def test_getPlayerHand(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        hand = opponent.get_player_hand()
        self.assertIn(hand, ['R', 'P', 'S'])


class TestBaseOpponentClearMoves(TestBaseOpponent):

    def test_outerPiecePlayerClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['O'][4] = 'PR'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('O', 4)
        self.assertEqual(possible_moves, ['O4:O5', 'O4:O3', 'O4:I2'])

    def test_innerPiecePlayerClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['I'][2] = 'PR'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('I', 2)
        self.assertEqual(possible_moves, ['I2:I3', 'I2:I1', 'I2:O4', 'I2:O5', 'I2:C0'])

    def test_centerPiecePlayerClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['C'][0] = 'PR'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('C', 0)
        self.assertEqual(possible_moves, [('C0:I%s' % i) for i in range(9)])

    def test_outerPieceOpponentClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['O'][4] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('O', 4)
        self.assertEqual(possible_moves, ['O4:O5', 'O4:O3', 'O4:I2'])

    def test_innerPieceOpponentClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['I'][2] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('I', 2)
        self.assertEqual(possible_moves, ['I2:I3', 'I2:I1', 'I2:O4', 'I2:O5', 'I2:C0'])

    def test_centerPieceOpponentClearMoves(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['C'][0] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('C', 0)
        self.assertEqual(possible_moves, [('C0:I%s' % i) for i in range(9)])


class TestBaseOpponentChallengeMoves(TestBaseOpponent):

    def test_outerPiecePlayerChallengeOuterMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['O'][0] = 'PR'
        board['O'][1] = 'PR'
        board['O'][-1] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('O', 0)
        self.assertEqual(possible_moves, ['O0:O17', 'O0:I0'])

    def test_outerPiecePlayerChallengeInnerMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['O'][0] = 'PR'
        board['O'][1] = 'PR'
        board['I'][0] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('O', 1)
        self.assertEqual(possible_moves, ['O1:O2', 'O1:I0'])

    def test_innerPiecePlayerChallengeOuterMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['I'][0] = 'PR'
        board['O'][1] = 'PR'
        board['O'][0] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('I', 0)
        self.assertEqual(possible_moves, ['I0:I1', 'I0:I8', 'I0:O0', 'I0:C0'])

    def test_innerPiecePlayerChallengeInnerMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['I'][0] = 'PR'
        board['O'][1] = 'PR'
        board['I'][8] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('I', 0)
        self.assertEqual(possible_moves, ['I0:I1', 'I0:I8', 'I0:O0', 'I0:C0'])

    def test_innerPiecePlayerChallengeCenterMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['I'][0] = 'PR'
        board['O'][1] = 'PR'
        board['C'][0] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('I', 0)
        self.assertEqual(possible_moves, ['I0:I1', 'I0:I8', 'I0:O0', 'I0:C0'])

    def test_centerPiecePlayerChallengeInnerMove(self):
        opponent = ConcreteBaseOpponent()
        board = opponent.board
        board['C'][0] = 'PR'
        board['I'][0] = 'PR'
        board['I'][1] = 'PS'
        board['I'][8] = 'OU'
        opponent.reset_board(board)
        possible_moves = opponent.get_move_list('C', 0)
        self.assertEqual(possible_moves, [('C0:I%s' % i) for i in range(2, 9)])


class TestBaseOpponentApplyMovePlayer(TestBaseOpponent):

    def test_applyPlayerMovement(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O0',
            'to': 'I0',
            'outcome': 'M'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][0])
        self.assertEqual('PR', opponent.board['I'][0])

    def test_applyPlayerChallengeTie(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O0',
            'to': 'O17',
            'outcome': 'T',
            'otherHand': 'R'
        }
        opponent.apply_move(move)
        self.assertEqual('PR!', opponent.board['O'][0])
        self.assertEqual('OR', opponent.board['O'][17])

    def test_applyPlayerChallengeWin(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O0',
            'to': 'O17',
            'outcome': 'W',
            'otherHand': 'S'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][0])
        self.assertEqual('PR!', opponent.board['O'][17])
        self.assertEqual(opponent.captures, [0, 0, 1])

    def test_applyPlayerChallengeLoss(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O0',
            'to': 'O17',
            'outcome': 'L',
            'otherHand': 'P'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][0])
        self.assertEqual('OP', opponent.board['O'][17])

    def test_applyPlayerSurrender(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'surrender': True
        }
        opponent.apply_move(move)
        self.assertEqual(self.DEFAULT_BLUE_BOARD, opponent.board)


class TestBaseOpponentApplyMoveOpponent(TestBaseOpponent):

    def test_applyOpponentMovement(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O17',
            'to': 'I8',
            'outcome': 'M'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][17])
        self.assertEqual('OU', opponent.board['I'][8])

    def test_applyOpponentChallengeTie(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O17',
            'to': 'O0',
            'outcome': 'T',
            'otherHand': 'R'
        }
        opponent.apply_move(move)
        self.assertEqual('PR!', opponent.board['O'][0])
        self.assertEqual('OR', opponent.board['O'][17])

    def test_applyOpponentChallengeWin(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O17',
            'to': 'O0',
            'outcome': 'W',
            'otherHand': 'P'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][17])
        self.assertEqual('OP', opponent.board['O'][0])

    def test_applyOpponentChallengeLoss(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'from': 'O17',
            'to': 'O0',
            'outcome': 'L',
            'otherHand': 'S'
        }
        opponent.apply_move(move)
        self.assertEqual('0', opponent.board['O'][17])
        self.assertEqual('PR!', opponent.board['O'][0])

    def test_applyOpponentSurrender(self):
        opponent = ConcreteBaseOpponent()
        opponent.init_board_layout(0)
        move = {
            'surrender': True
        }
        opponent.apply_move(move)
        self.assertEqual(self.DEFAULT_BLUE_BOARD, opponent.board)
