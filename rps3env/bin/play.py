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
import argparse
import os

import gym
import numpy as np
import pyglet
from numpy.linalg import norm
from pyglet import gl
from pyglet.window import mouse, key

# noinspection PyUnresolvedReferences
import rps3env
from rps3env import envs, config
from rps3env.envs.rps3_game import BOARD_POSITIONS

FILE_DIR = os.path.dirname(__file__)
BACKGROUND_FILENAME = os.path.join(FILE_DIR, '../assets/board.png')
EMPTY_CIRCLE_FILENAME = os.path.join(FILE_DIR, '../assets/circle_empty.png')
FILLED_CIRCLE_FILENAME = os.path.join(FILE_DIR, '../assets/circle_filled.png')

__author__ = 'Islam Elnabarawy'

SELECTION_RADIUS = 30

BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50


def draw_circle(img: pyglet.image, x: int, y: int):
    img.blit(x + BOARD_OFFSET_X, y + BOARD_OFFSET_Y, width=SELECTION_RADIUS * 2, height=SELECTION_RADIUS * 2)


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


def find_cell(x, y, start=0, stop=9, skip=None):
    found_index = None
    found_point = None
    # find the point being selected
    for i in range(start, stop):
        if skip is not None and i in skip:
            continue
        ring, index = i2l(i)
        x_cell = BOARD_POSITIONS[ring][index][0] + BOARD_OFFSET_X
        y_cell = BOARD_POSITIONS[ring][index][1] + BOARD_OFFSET_Y
        if within_radius(x, y, x_cell, y_cell):
            found_index = i
            found_point = (x, y)
            break
    return found_index, found_point


def within_radius(x_1, y_1, x_2, y_2):
    return abs(norm(np.array((x_1, y_1)) - np.array((x_2, y_2)))) <= SELECTION_RADIUS
    # return (abs(x_1 - x_2) <= SELECTION_RADIUS) and (abs(y_1 - y_2) <= SELECTION_RADIUS)


class RPS3Game(object):
    def __init__(self, difficulty=2, random_seed=None) -> None:
        if difficulty <= 0:
            self.env = gym.make('RPS3Game-v0')  # type: envs.RPS3GameEnv
        else:
            self.env = gym.make('RPS3Game-v1')  # type: envs.RPS3GameMinMaxEnv
            self.env.settings['depth_limit'] = difficulty
        self.env.seed(random_seed)
        self.obs = self.env.reset()
        self.game_over = False
        self.last_reward = [0, 0]
        self.info = {}

        self.game_started = False
        self.current_selection = None
        self.current_point = None
        self.initial_setup = [1, 2, 3] * 3

        self.window = pyglet.window.Window(width=config.VIEWER_WIDTH, height=config.VIEWER_HEIGHT)
        self.bg = pyglet.image.load(BACKGROUND_FILENAME)  # type: pyglet.image.Texture
        self.circle_empty = pyglet.image.load(EMPTY_CIRCLE_FILENAME)
        self.circle_empty.anchor_x = SELECTION_RADIUS
        self.circle_empty.anchor_y = SELECTION_RADIUS
        self.circle_filled = pyglet.image.load(FILLED_CIRCLE_FILENAME)
        self.circle_filled.anchor_x = SELECTION_RADIUS
        self.circle_filled.anchor_y = SELECTION_RADIUS

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.window.on_draw = self._on_draw
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_release = self._on_key_release
        self.window.on_close = self._on_close

    def _on_draw(self):
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()

        if not self.game_started:
            self._draw_board_setup()
        else:
            self._draw_match()

        if self.game_over:
            self._draw_game_over()

        self.window.flip()

    def _draw_board_setup(self):
        bg = self.bg.get_region(0, 0, width=self.bg.width, height=self.bg.height // 2)
        bg.blit(BOARD_OFFSET_X, BOARD_OFFSET_Y, width=600, height=300)

        pyglet.text.Label(
            "Drag and drop to order your pieces.",
            font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
            x=350, y=550, color=(255, 0, 0, 255)
        ).draw()
        pyglet.text.Label(
            "Press Enter to start match.",
            font_name='Arial', font_size=28, anchor_x='center', anchor_y='center',
            x=350, y=500, color=(255, 0, 0, 255)
        ).draw()

        for index in range(9):
            x, y = BOARD_POSITIONS['O'][index]
            if self.current_selection is not None:
                if self.current_selection == index:
                    continue
                else:
                    if within_radius(x + BOARD_OFFSET_X, y + BOARD_OFFSET_Y,
                                     self.current_point[0], self.current_point[1]):
                        draw_circle(self.circle_filled, x, y)
                    else:
                        draw_circle(self.circle_empty, x, y)
            draw_piece(x, y, self.initial_setup[index], True)

        if self.current_selection is not None:
            x, y = self.current_point
            draw_piece(x - BOARD_OFFSET_X, y - BOARD_OFFSET_Y, self.initial_setup[self.current_selection], True)

    def _draw_match(self):
        self.bg.blit(BOARD_OFFSET_X, BOARD_OFFSET_Y, width=600, height=600)

        if 'round' in self.info and self.info['round'] > 0:
            txt = "Round {}: {} > {}".format(self.info['round'], *self.info['player_move'])
            if self.info['opponent_move'] is not None:
                txt += ", {} > {}".format(*self.info['opponent_move'])
            else:
                txt += ' - Game Over.'
            pyglet.text.Label(
                txt, font_name='Arial', font_size=16, anchor_x='left', anchor_y='center',
                x=10, y=config.VIEWER_HEIGHT - 30, color=(0, 0, 0, 255)
            ).draw()

        available_spots = [x2 for x1, x2 in self.env.available_actions if x1 == self.current_selection] \
            if not self.game_over else []
        for index in [i for i in range(28) if self.obs['occupied'][i] or (i in available_spots)]:
            r, i = i2l(index)
            x, y = BOARD_POSITIONS[r][i]
            if self.current_selection is not None:
                if self.current_selection == index:
                    continue
                elif index in available_spots:
                    if within_radius(x + BOARD_OFFSET_X, y + BOARD_OFFSET_Y,
                                     self.current_point[0], self.current_point[1]):
                        draw_circle(self.circle_filled, x, y)
                    else:
                        draw_circle(self.circle_empty, x, y)
            if self.obs['occupied'][index]:
                draw_piece(x, y, self.obs['piece_type'][index], self.obs['player_owned'][index])

        if self.current_selection is not None:
            x, y = self.current_point
            draw_piece(x - BOARD_OFFSET_X, y - BOARD_OFFSET_Y, self.obs['piece_type'][self.current_selection], True)

    def _draw_game_over(self):
        txt = "Game Over! {} won.".format('Player' if sum(self.last_reward) > 0 else 'Opponent')
        pyglet.text.Label(
            txt, font_name='Arial', font_size=32, anchor_x='center', anchor_y='center',
            x=350, y=30, color=(0, 0, 0, 255)
        ).draw()

    def _on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if not self.game_started:
            found_index, found_point = find_cell(x, y)
            self.current_selection = found_index
            self.current_point = found_point
        elif not self.game_over:
            skip = [i for i in range(28) if not self.obs['player_owned'][i]]
            found_index, found_point = find_cell(x, y, stop=28, skip=skip)
            self.current_selection = found_index
            self.current_point = found_point

    def _on_mouse_release(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        if self.current_selection is not None:
            if not self.game_started:
                # check to see if the piece was dropped over another piece
                found_index, found_point = find_cell(x, y, skip=[self.current_selection])
                if found_index is not None:
                    # swap the two pieces
                    self.initial_setup[found_index], self.initial_setup[self.current_selection] = \
                        self.initial_setup[self.current_selection], self.initial_setup[found_index]
            else:
                # check to see if the piece was dropped over a legal spot
                available_spots = [x2 for x1, x2 in self.env.available_actions if x1 == self.current_selection]
                skip = [i for i in range(28) if i not in available_spots]
                found_index, found_point = find_cell(x, y, stop=28, skip=skip)
                if found_index is not None:
                    self.obs, self.last_reward, self.game_over, self.info = \
                        self.env.step((self.current_selection, found_index))
            self.current_selection = None
            self.current_point = None

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not buttons & mouse.LEFT:
            return
        if self.current_selection is not None:
            self.current_point = (x, y)

    def _on_key_release(self, symbol, modifiers):
        if not self.game_started and symbol in [key.ENTER, key.RETURN]:
            self.obs, self.last_reward, self.game_over, self.info = self.env.step(self.initial_setup)
            self.game_started = True

    def _on_close(self):
        self.env.close()
        self.window.close()

    def run(self):
        pyglet.app.run()


def main():
    parser = argparse.ArgumentParser(description='Play through the game environment using human control.')
    parser.add_argument("--difficulty", type=int, default=2, choices=range(10), help="Difficulty level; 0 is random.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for the random number generator.")
    args = parser.parse_args()

    RPS3Game(args.difficulty, args.random_seed).run()


if __name__ == '__main__':
    main()
