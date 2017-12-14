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

import gym
import pyglet
from pyglet import gl
from pyglet.window import mouse

# noinspection PyUnresolvedReferences
import rps3env
from rps3env import envs, config
from rps3env.envs.rps3_game import BOARD_POSITIONS

__author__ = 'Islam Elnabarawy'

SELECTION_RADIUS = 60

BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50


def draw_circle(img: pyglet.image, x: int, y: int):
    img.blit(x + BOARD_OFFSET_X, y + BOARD_OFFSET_Y, width=SELECTION_RADIUS, height=SELECTION_RADIUS)


def i2l(i):
    if i < 18: return 'O', i
    if i < 27: return 'I', i - 18
    return 'C', 0


def draw_piece(x, y, piece_type, player_owned):
    color = (0, 0, 255, 255) if player_owned else (255, 0, 0, 255)
    text = '?RPS'[piece_type]
    pyglet.text.Label(
        text, font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
        x=x + BOARD_OFFSET_X, y=y + BOARD_OFFSET_Y, color=color
    ).draw()


class RPS3Game(object):
    def __init__(self) -> None:
        self.env = gym.make('RPS3Game-v1')  # type: envs.RPS3GameEnv
        self.env.seed(0)
        self.obs = self.env.reset()
        self.game_over = False
        self.last_reward = [0, 0]
        self.info = {}

        self.game_started = False
        self.current_selection = None
        self.current_point = None
        self.initial_setup = [1, 2, 3] * 3

        self.window = pyglet.window.Window(width=config.VIEWER_WIDTH, height=config.VIEWER_HEIGHT)
        self.bg = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/board.png'))
        self.circle = pyglet.image.load(os.path.join(os.path.dirname(__file__), '../assets/circle.png'))
        self.circle.anchor_x = SELECTION_RADIUS // 2
        self.circle.anchor_y = SELECTION_RADIUS // 2

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.window.on_draw = self._on_draw
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_close = self._on_close

    def _on_draw(self):
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()

        self.bg.blit(BOARD_OFFSET_X, BOARD_OFFSET_Y, width=600, height=600)

        if not self.game_started:
            for index in range(9):
                x, y = BOARD_POSITIONS['O'][index]
                if self.current_selection is not None:
                    if self.current_selection == index:
                        continue
                    else:
                        draw_circle(self.circle, x, y)
                draw_piece(x, y, self.initial_setup[index], True)
            if self.current_selection is not None:
                x, y = self.current_point
                draw_piece(x - BOARD_OFFSET_X, y - BOARD_OFFSET_Y, self.initial_setup[self.current_selection], True)
        else:
            for index in [i for i in range(28) if self.obs['occupied'][i]]:
                ring, index = i2l(index)
                draw_piece(ring, index, self.obs['piece_type'][index], self.obs['player_owned'][index])

        if self.game_over:
            txt = "Game Over! {} won.".format('Player' if sum(self.last_reward) > 0 else 'Opponent')
            pyglet.text.Label(
                txt, font_name='Arial', font_size=32, anchor_x='center', anchor_y='center',
                x=350, y=30, color=(0, 0, 0, 255)
            ).draw()

        self.window.flip()

    def _on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if not self.game_started:
            # find the point being selected
            for i in range(9):
                ring, index = i2l(i)
                x_cell, y_cell = BOARD_POSITIONS[ring][index]
                x_cell += BOARD_OFFSET_X
                y_cell += BOARD_OFFSET_Y
                if abs(x - x_cell) <= SELECTION_RADIUS and abs(y - y_cell) <= SELECTION_RADIUS:
                    self.current_selection = i
                    self.current_point = (x, y)
                    return
        else:
            pass

    def _on_mouse_release(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if not self.game_started:
            self.current_selection = None
            self.current_point = None

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not buttons & mouse.LEFT:
            return
        if self.current_selection is not None:
            self.current_point = (x, y)

    def _on_close(self):
        self.env.close()
        self.window.close()

    def run(self):
        pyglet.app.run()


def main():
    RPS3Game().run()


if __name__ == '__main__':
    main()
