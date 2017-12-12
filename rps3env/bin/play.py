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

import os
import random

import gym
import pyglet
from pyglet import gl
from pyglet.window import mouse, key

# noinspection PyUnresolvedReferences
import rps3env
from rps3env import envs
from rps3env.classes import BoardLocation
from rps3env.envs.rps3_game import BOARD_POSITIONS

SELECTION_RADIUS = 50

__author__ = 'Islam Elnabarawy'


class PlayGame(object):
    def __init__(self) -> None:
        self.env = gym.make('RPS3Game-v1')  # type: envs.RPS3GameEnv
        self.env.seed(0)
        self.env.reset()

        self.window = pyglet.window.Window(width=rps3env.config.VIEWER_WIDTH, height=rps3env.config.VIEWER_HEIGHT)
        self.bg = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/board.png'))
        self.circle = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/circle.png'))

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.window.on_draw = self._on_draw
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_key_release = self._on_key_release

    def _on_draw(self):
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()

        board_offset_x = 50
        board_offset_y = 50
        self.bg.blit(board_offset_x, board_offset_y, width=600, height=600)

        # for r in 'OIC':
        #     for i, p in enumerate(BOARD_POSITIONS[r]):
        #         x, y = p
        #         self.circle.blit(x + SELECTION_RADIUS // 2, y + SELECTION_RADIUS // 2,
        #                          width=SELECTION_RADIUS, height=SELECTION_RADIUS)

        def draw_location(location: BoardLocation):
            assert location.piece is not None
            x, y = BOARD_POSITIONS[location.ring][location.index]
            color = (0, 0, 255, 255) if location.piece.player_owned else (255, 0, 0, 255)
            text = ('{}!' if location.piece.player_owned and location.piece.revealed else '{}').format(
                location.piece.to_str(True))
            pyglet.text.Label(
                text, font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
                x=x + board_offset_x, y=y + board_offset_y, color=color
            ).draw()

        for l in [location for location in self.env._board['O'] + self.env._board['I'] + self.env._board['C']
                  if location.piece is not None]:
            draw_location(l)

        if self.env._game_over:
            txt = "Game Over! {} won.".format('Player' if self.env._player_won else 'Opponent')
            pyglet.text.Label(
                txt, font_name='Arial', font_size=32, anchor_x='center', anchor_y='center',
                x=350, y=30, color=(0, 0, 0, 255)
            ).draw()

        self.window.flip()

    def _on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            print('Mouse press at', (x, y))

    def _on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            print('Mouse release at', (x, y))

    def _on_key_release(self, symbol, modifiers):
        if symbol == key.ENTER:
            if self.env._game_over:
                self.env.reset()
            self.env.step(random.choice(self.env.available_actions))

    def run(self):
        pyglet.app.run()


def main():
    PlayGame().run()


if __name__ == '__main__':
    main()
