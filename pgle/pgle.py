import numpy as np
import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
from .games.base.pygamewrapper import PyGameWrapper
from pygame.constants import K_w, K_a, K_s, K_d, K_SPACE

class PGLE(object):
    def __init__(self, game):

        self.game = game
        self.rng = np.random.RandomState(24)
        self.game.setRNG(self.rng)
        # self.init()
        self.NOOP = None
        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.frame_count = 0
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.game.setRNG(self.rng)
        self.game._setup()
        self.game.init()
        self.name = self.game.__class__.__name__
        self.actions = None
        self.initActionSet()
        self.actions2name = {
            K_w:"up",
            K_a:"left",
            K_d:"right",
            K_s:"down",
            K_SPACE:"fire",
        }

    def init(self):
        """
        Initializes the game. This depends on the game and could include
        doing things such as setting up the display, clock etc.
        This method should be explicitly called.
        """
        self.game._setup()
        self.game.init() #this is the games setup/init

    def getActionSet(self):
        """
        Gets the actions the game supports. Optionally inserts the NOOP
        action if PLE has add_noop_action set to True.
        Returns
        --------
        list of pygame.constants
            The agent can simply select the index of the action
            to perform.
        """
        return self.actions

    def initActionSet(self):
        """
        Gets the actions the game supports. Optionally inserts the NOOP
        action if PLE has add_noop_action set to True.
        Returns
        --------
        list of pygame.constants
            The agent can simply select the index of the action
            to perform.
        """
        actions = self.game.actions
        if (sys.version_info > (3, 0)): #python ver. 3
            if isinstance(actions, dict) or isinstance(actions, dict_values):
                actions = actions.values()
        else:
            if isinstance(actions, dict):
                actions = actions.values()

        actions = sorted(list(actions)) #.values()
        #print (actions)
        #assert isinstance(actions, list), "actions is not a list"

        actions.append(self.NOOP)
        self.actions = actions

    def getActionName(self, action):
        if self.actions[action] == self.NOOP:
            return "noop"
        else:
            return self.actions2name[self.actions[action]]

    def getFrameNumber(self):
        """
        Gets the current number of frames the agent has seen
        since PLE was initialized.
        Returns
        --------
        int
        """

        return self.frame_count

    def game_over(self):
        """
        Returns True if the game has reached a terminal state and
        False otherwise.
        This state is game dependent.
        Returns
        -------
        bool
        """

        return self.game.game_over()

    def score(self):
        """
        Gets the score the agent currently has in game.
        Returns
        -------
        int
        """

        return self.game.getScore()

    def reset(self):
        """
        Performs a reset of the games to a clean initial state.
        """
        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.game.reset()
        return self.getGameState()

    def render(self):
        """
        Gets the current game screen in RGB format.
        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).
        """

        return self.game.getScreenRGB()

    def getScreenDims(self):
        """
        Gets the games screen dimensions.
        Returns
        -------
        tuple of int
            Returns a tuple of the following format (screen_width, screen_height).
        """
        return self.game.getScreenDims()

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.
        This can include items such as player position, velocity, ball location and velocity etc.
        Returns
        -------
        dict
            It returns a set of local information and a dict of global information.
        """
        return self.game.getGameState()
    
    def getEnvState(self):
        state = {"state":self.getGameState(),
                 "rng":self.rng,
                 "last_action":self.last_action,
                 "action":self.action,
                 "previous_score":self.previous_score,
                 "frame_count":self.frame_count,
                 }
        return state
        
    def loadEnvState(self, state):
        self.rng = state["rng"]
        self.last_action = state["last_action"]
        self.action = state["action"]
        self.previous_score = state["previous_score"]
        self.frame_count = state["frame_count"]
        self.game.loadGameState(state["state"])

    def act(self, action):
        """
        Performs an action on the game. Checks if the game is over or if the provided action is valid based on the allowed action set.
        """
        if self.game_over():
            return 0.0

        if action not in self.getActionSet():
            action = self.NOOP

        self._setAction(action)
        if self.name[-4:] == "Maze":
            for i in range(self.game.fps):
                self.game.step()
        else:
            self.game.step(1000 / self.game.fps)
        
        self._draw_frame()

        self.frame_count += 1

        return self._getReward()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        _action = self.getActionSet()[action]
        reward = self.act(_action)
        return self.getGameState(), reward, self.game_over(), {}


    def _draw_frame(self):
        """
        Decides if the screen will be drawn too
        """

        self.game._draw_frame(False)

    def _setAction(self, action):
        """
            Instructs the game to perform an action if its not a NOOP
        """

        if action is not None:
            self.game._setAction(action, self.last_action)
        else:
            self.game.player.vel.x = 0
            self.game.player.vel.y = 0
        self.last_action = action

    def _getReward(self):
        """
        Returns the reward the agent has gained as the difference between the last action and the current one.
        """
        reward = self.game.getScore() - self.previous_score
        self.previous_score = self.game.getScore()

        return reward