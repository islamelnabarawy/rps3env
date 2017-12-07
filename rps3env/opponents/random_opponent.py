import random

from rps3env.opponents import BaseOpponent


class RandomOpponent(BaseOpponent):

    def get_board_layout(self):
        layout = ['R', 'P', 'S'] * 3
        random.shuffle(layout)
        return layout

    def get_player_hand(self):
        hand = random.choice(['R', 'P', 'S'])
        return hand

    def get_next_move(self):
        choices = self.get_possible_moves('P')
        return random.choice(choices) if len(choices) > 0 else None
