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

from rps3env import classes

__author__ = 'Islam Elnabarawy'

BLUE_SETUP = [1, 2, 3, 1, 2, 3, 1, 2, 3]
RED_SETUP = [3, 1, 2, 3, 1, 2, 3, 1, 2]
BAD_SETUP = [1, 2, 3, 3, 3, 3, 2, 2, 1]


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.match = classes.Match()

    def setup_board(self):
        self.match.set_board(BLUE_SETUP, classes.PlayerColor.Blue)
        self.match.set_board(RED_SETUP, classes.PlayerColor.Red)

    def test_init(self):
        self.assertIsNotNone(self.match)

    def test_empty_board(self):
        self.assertListEqual(self.match.board, [None for _ in range(28)])

    def test_valid_setup(self):
        self.match.set_board(BLUE_SETUP, classes.PlayerColor.Blue)
        for i in range(9):
            self.assertEqual(classes.PlayerColor.Blue, self.match.board[i].color)
            self.assertEqual(BLUE_SETUP[i], self.match.board[i].piece_type.value)
        for i in range(9, 28):
            self.assertIsNone(self.match.board[i])
        self.match.set_board(RED_SETUP, classes.PlayerColor.Red)
        for i in range(9, 18):
            self.assertEqual(classes.PlayerColor.Red, self.match.board[i].color)
            self.assertEqual(RED_SETUP[i - 9], self.match.board[i].piece_type.value)
        for i in range(18, 28):
            self.assertIsNone(self.match.board[i])

    def test_invalid_setup(self):
        self.assertRaises(AssertionError, lambda: self.match.set_board(BAD_SETUP, classes.PlayerColor.Blue))
        self.assertRaises(AssertionError, lambda: self.match.set_board(BAD_SETUP, classes.PlayerColor.Red))

    def test_double_setup(self):
        self.match.set_board(BLUE_SETUP, classes.PlayerColor.Blue)
        self.assertRaises(AssertionError, lambda: self.match.set_board(BLUE_SETUP, classes.PlayerColor.Blue))
        self.match.set_board(RED_SETUP, classes.PlayerColor.Red)
        self.assertRaises(AssertionError, lambda: self.match.set_board(RED_SETUP, classes.PlayerColor.Red))

    def test_make_move_too_soon_blue(self):
        self.match.set_board(BLUE_SETUP, classes.PlayerColor.Blue)
        self.assertRaises(AssertionError, lambda: self.match.make_move(0, 18, classes.PlayerColor.Blue))

    def test_make_move_too_soon_red(self):
        self.match.set_board(RED_SETUP, classes.PlayerColor.Red)
        self.assertRaises(AssertionError, lambda: self.match.make_move(9, 22, classes.PlayerColor.Red))

    def test_make_move_legal_movement(self):
        self.setup_board()
        piece_to_move = self.match.board[0]
        self.assertEqual((0, None), self.match.make_move(0, 18, classes.PlayerColor.Blue))
        self.assertIsNone(self.match.board[0])
        self.assertEqual(piece_to_move, self.match.board[18])

    def test_make_move_illegal_movements(self):
        self.setup_board()
        self.assertRaises(AssertionError, lambda: self.match.make_move(0, 19, classes.PlayerColor.Blue))
        self.assertRaises(AssertionError, lambda: self.match.make_move(8, 21, classes.PlayerColor.Blue))

    def test_make_move_twice(self):
        self.setup_board()
        self.assertEqual((0, None), self.match.make_move(0, 18, classes.PlayerColor.Blue))
        self.assertRaises(AssertionError, lambda: self.match.make_move(18, 0, classes.PlayerColor.Blue))
        self.assertEqual((0, None), self.match.make_move(9, 22, classes.PlayerColor.Red))
        self.assertRaises(AssertionError, lambda: self.match.make_move(22, 9, classes.PlayerColor.Red))

    def test_make_move_wrong_turn(self):
        self.setup_board()
        self.assertRaises(AssertionError, lambda: self.match.make_move(9, 22, classes.PlayerColor.Red))

    def test_make_move_wrong_piece(self):
        self.setup_board()
        self.assertRaises(AssertionError, lambda: self.match.make_move(9, 22, classes.PlayerColor.Blue))

    def test_make_move_no_piece(self):
        self.setup_board()
        self.assertRaises(AssertionError, lambda: self.match.make_move(18, 19, classes.PlayerColor.Blue))
        self.assertRaises(AssertionError, lambda: self.match.make_move(27, 18, classes.PlayerColor.Blue))

    def test_make_move_challenge_tie(self):
        self.setup_board()
        self.assertEqual((0, classes.PieceType.S), self.match.make_move(8, 9, classes.PlayerColor.Blue))
        self.assertTrue(self.match.board[8].revealed)
        self.assertTrue(self.match.board[9].revealed)

    def test_make_move_challenge_win(self):
        self.setup_board()
        self.match.make_move(8, 9, classes.PlayerColor.Blue)
        self.assertEqual((1, classes.PieceType.R), self.match.make_move(17, 0, classes.PlayerColor.Red))
        self.assertIsNone(self.match.board[17])
        self.assertTrue(self.match.board[0].revealed)

    def test_make_move_challenge_loss(self):
        self.setup_board()
        self.assertEqual((-1, classes.PieceType.P), self.match.make_move(0, 17, classes.PlayerColor.Blue))
        self.assertIsNone(self.match.board[0])
        self.assertTrue(self.match.board[17].revealed)

    def test_make_move_challenge_win_sequence(self):
        self.setup_board()
        move_sequence = [
            (8, 22, classes.PlayerColor.Blue), (11, 23, classes.PlayerColor.Red), (22, 23, classes.PlayerColor.Blue)
        ]
        result_sequence = [
            (0, None), (0, None), (1, classes.PieceType.P)
        ]
        while len(move_sequence) > 0:
            self.assertEqual(result_sequence.pop(0), self.match.make_move(*move_sequence.pop(0)))
        self.assertIsNone(self.match.board[8])
        self.assertIsNone(self.match.board[11])
        self.assertIsNone(self.match.board[22])
        self.assertTrue(self.match.board[23].revealed)
        self.assertEqual(classes.PieceType.S, self.match.board[23].piece_type)

    def test_make_move_challenge_loss_sequence(self):
        self.setup_board()
        move_sequence = [
            (2, 19, classes.PlayerColor.Blue), (17, 26, classes.PlayerColor.Red),
            (19, 18, classes.PlayerColor.Blue), (26, 18, classes.PlayerColor.Red)
        ]
        result_sequence = [
            (0, None), (0, None), (0, None), (-1, classes.PieceType.S)
        ]
        while len(move_sequence) > 0:
            self.assertEqual(result_sequence.pop(0), self.match.make_move(*move_sequence.pop(0)))
        self.assertIsNone(self.match.board[2])
        self.assertIsNone(self.match.board[17])
        self.assertTrue(self.match.board[18].revealed)
        self.assertEqual(classes.PieceType.S, self.match.board[18].piece_type)

    def test_clone(self):
        self.setup_board()
        match_clone = self.match.clone()
        for i, p in enumerate(match_clone.board):
            if p is not None:
                original_piece = self.match.board[i]  # type: classes.BoardPiece
                cloned_piece = match_clone.board[i]  # type: classes.BoardPiece
                self.assertNotEqual(cloned_piece, original_piece)
                self.assertEqual(cloned_piece.piece_type, original_piece.piece_type)
                self.assertEqual(cloned_piece.color, original_piece.color)
                self.assertEqual(cloned_piece.revealed, original_piece.revealed)
            else:
                self.assertIsNone(self.match.board[i])
        self.match.make_move(0, 18, classes.PlayerColor.Blue)
        self.assertNotEqual(match_clone._round, self.match._round)
        self.assertNotEqual(match_clone._moves, self.match._moves)
        self.assertIsNotNone(match_clone.board[0])
        self.assertIsNone(match_clone.board[18])

    def test_possible_moves(self):
        self.setup_board()

        possible_moves = [
            (0, 17), (0, 18), (1, 18), (2, 19), (3, 19), (4, 20), (5, 20), (6, 21), (7, 21), (8, 9), (8, 22)
        ]

        color, moves = self.match.get_possible_moves()
        self.assertEqual(classes.PlayerColor.Blue, color)
        self.assertEqual(possible_moves, moves)

        self.match.make_move(0, 18, classes.PlayerColor.Blue)

        possible_moves = [
            (9, 8), (9, 22), (10, 23), (11, 23), (12, 24), (13, 24), (14, 25), (15, 25), (16, 26), (17, 0), (17, 26)
        ]

        color, moves = self.match.get_possible_moves()
        self.assertEqual(classes.PlayerColor.Red, color)
        self.assertEqual(possible_moves, moves)
