import pgle
import sys
if len(sys.argv) >= 2:
    game_name = sys.argv[1]
    pgle.play(game_name)
else:
    raise Exception('Please input a game name')
