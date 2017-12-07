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
import random

from rps3env.opponents import BaseOpponent

__author__ = 'Islam Elnabarawy'


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
