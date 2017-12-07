from rps3env.opponents import MinMaxOpponent
from rps3env.tests.base_opponent_test import TestBaseOpponent


class TestMinMaxOpponentLegalNextMove(TestBaseOpponent):

    def test_minMaxLegalBlueMove(self):
        opponent = MinMaxOpponent(2)
        opponent.init_board_layout(0)
        move = opponent.get_next_move()
        self.assertIn(move, self.POSSIBLE_BLUE_MOVES)

    def test_minMaxLegalGreenMove(self):
        opponent = MinMaxOpponent(2)
        opponent.init_board_layout(1)
        move = opponent.get_next_move()
        self.assertIn(move, self.POSSIBLE_GREEN_MOVES)


class TestMinMaxOpponentNextMove(TestBaseOpponent):

    def test_minMaxSingleLevelBlueMove(self):
        opponent = MinMaxOpponent(1)
        layout = ['R', 'P', 'S'] * 3
        opponent.init_board_layout(0, layout)
        move = opponent.get_next_move()
        self.assertEqual('O0:O17', move)

    def test_minMaxTwoLevelsBlueMove(self):
        opponent = MinMaxOpponent(2)
        layout = ['R', 'P', 'S'] * 3
        opponent.init_board_layout(0, layout)
        move = opponent.get_next_move()
        self.assertEqual('O0:O17', move)


class TestMinMaxOpponentLegalBoardLayout(TestBaseOpponent):

    def test_minMaxLegalBlueLayout(self):
        opponent = MinMaxOpponent()
        opponent.init_board_layout(0)
        layout = opponent.board['O'][:9]
        self.assertEqual(layout.count('PR'), 3)
        self.assertEqual(layout.count('PP'), 3)
        self.assertEqual(layout.count('PS'), 3)

    def test_minMaxLegalGreenLayout(self):
        opponent = MinMaxOpponent()
        opponent.init_board_layout(1)
        layout = opponent.board['O'][9:]
        self.assertEqual(layout.count('PR'), 3)
        self.assertEqual(layout.count('PP'), 3)
        self.assertEqual(layout.count('PS'), 3)


class TestMinMaxOpponentLegalPlayerHand(TestBaseOpponent):

    def test_minMaxLegalPlayerHand(self):
        opponent = MinMaxOpponent()
        opponent.init_board_layout(0)
        hand = opponent.get_player_hand()
        self.assertIn(hand, ['R', 'P', 'S'])
