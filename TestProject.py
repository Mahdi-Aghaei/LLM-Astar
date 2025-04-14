import sys
sys.path.append('.')
from llmastar.pather import AStar, LLMAStar

query = {
    "start": [10, 5], "goal": [10, 15], "size": [51, 31],
    "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
    "vertical_barriers": [[25, 10, 22]],
    "range_x": [0, 51], "range_y": [0, 31]
}

astar = AStar().searching(query=query, filepath='astar.png')

llm = LLMAStar(llm='llama', prompt='standard').searching(query=query, filepath='llm.png')