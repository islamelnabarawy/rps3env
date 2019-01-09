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
from gym.envs.registration import register

__author__ = 'Islam Elnabarawy'


register(
    id='RPS3Game-v0',
    entry_point='rps3env.envs:RPS3GameEnv',
    kwargs={}
)

register(
    id='RPS3Game-v1',
    entry_point='rps3env.envs:RPS3GameMinMaxEnv',
    kwargs={
        'depth_limit': 2
    }
)
