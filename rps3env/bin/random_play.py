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
import argparse
import random
# noinspection PyUnresolvedReferences
import time

import gym

from rps3env import envs

__author__ = 'Islam Elnabarawy'


def main():
    parser = argparse.ArgumentParser(description='Play through the game environment using random moves.')
    parser.add_argument("--difficulty", type=int, default=2, choices=range(10), help="Difficulty level; 0 is random.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for the random number generator.")
    args = parser.parse_args()

    if args.difficulty <= 0:
        env = gym.make('RPS3Game-v0')  # type: envs.RPS3GameEnv
    else:
        env = gym.make('RPS3Game-v1')  # type: envs.RPS3GameMinMaxEnv
        env.settings['depth_limit'] = args.difficulty

    env.seed(args.random_seed)
    env.reset()

    action = random.choice(env.available_actions)
    obs, reward, done, info = env.step(action)

    while not done:
        env.render()
        action = random.choice(env.available_actions)
        obs, reward, done, info = env.step(action)

    env.render()
    time.sleep(3)
    env.close()


if __name__ == '__main__':
    main()
