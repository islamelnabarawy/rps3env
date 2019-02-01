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

__author__ = "Islam Elnabarawy"


class GameState(object):
    """
    Game state representation for search algorithms
    """

    def __init__(self):
        raise NotImplementedError()

    def get_possible_actions(self):
        """
        Get a list of all possible actions in the current state.

        :return: A list of Actions that are legal next actions in this state
        :rtype: list[Action]
        """
        raise NotImplementedError()

    def take_action(self, action):
        """
        Apply the given Action to a copy of the current state

        :type action: Action
        :rtype: GameState
        """
        raise NotImplementedError()

    def is_terminal(self):
        """
        Get whether the current state is a terminal state

        :return: A boolean indicating whether this is a terminal state
        :rtype: bool
        """
        raise NotImplementedError()

    def get_reward(self):
        """
        Get the final reward for a terminal state

        :return: The final reward for this state, assuming it is a terminal state
        :rtype: int
        """
        # only needed for terminal states
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return 'GameState(\n\t{}\n)'.format(',\n\t'.join([
            '{}={}'.format(k, self.__dict__[k])
            for k in sorted(self.__dict__.keys())
        ]))


class Action(object):
    def __init__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        """
        :type other: Action
        """
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()
