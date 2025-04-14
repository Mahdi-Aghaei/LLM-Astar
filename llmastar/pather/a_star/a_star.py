import heapq
import math
from llmastar.env.search import env, plotting
from llmastar.utils import is_lines_collision

class AStar:
    def __init__(self):
        pass

    def searching(self, query, filepath='temp.png', use_bidirectional=False):
        self.filepath = filepath
        self.s_start = tuple(query['start'])
        self.s_goal = tuple(query['goal'])
        self.horizontal_barriers = query['horizontal_barriers']
        self.vertical_barriers = query['vertical_barriers']
        self.range_x = query['range_x']
        self.range_y = query['range_y']
        self.Env = env.Env(self.range_x[1], self.range_y[1], self.horizontal_barriers, self.vertical_barriers)
        self.plot = plotting.Plotting(self.s_start, self.s_goal, self.Env)
        self.range_x[1] -= 1
        self.range_y[1] -= 1
        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        if use_bidirectional:
            path = self.bidirectional_search(self.s_start, self.s_goal)
            visited = []  # Optional: track visited nodes
        else:
            path, visited, operation, g = self.unidirectional_search()

        result = {
            "operation": len(visited) if not use_bidirectional else None,
            "storage": len(path) if not use_bidirectional else None,
            "length": sum(self._euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
        }
        self.plot.animation(path, visited, True, "A* Final", self.filepath)
        return result

    def bidirectional_search(self, start, goal):
        open_start = []
        open_goal = []
        heapq.heappush(open_start, (0, start))
        heapq.heappush(open_goal, (0, goal))

        came_from_start = {start: None}
        came_from_goal = {goal: None}
        g_score_start = {start: 0}
        g_score_goal = {goal: 0}
        best_path_cost = float('inf')
        best_meeting_point = None

        while open_start and open_goal:
            _, current_start = heapq.heappop(open_start)
            _, current_goal = heapq.heappop(open_goal)

            for neighbor in self.get_neighbor(current_start):
                tentative_g = g_score_start[current_start] + self.cost(current_start, neighbor)
                if neighbor not in g_score_start or tentative_g < g_score_start[neighbor]:
                    came_from_start[neighbor] = current_start
                    g_score_start[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_start, (f_score, neighbor))
                    if neighbor in g_score_goal:
                        total_cost = tentative_g + g_score_goal[neighbor]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            best_meeting_point = neighbor

            for neighbor in self.get_neighbor(current_goal):
                tentative_g = g_score_goal[current_goal] + self.cost(current_goal, neighbor)
                if neighbor not in g_score_goal or tentative_g < g_score_goal[neighbor]:
                    came_from_goal[neighbor] = current_goal
                    g_score_goal[neighbor] = tentative_g
                    f_score = tentative_g + self._euclidean_distance(neighbor, start)
                    heapq.heappush(open_goal, (f_score, neighbor))
                    if neighbor in g_score_start:
                        total_cost = tentative_g + g_score_start[neighbor]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            best_meeting_point = neighbor

            if best_meeting_point:
                path_start = self.reconstruct_path(came_from_start, best_meeting_point)
                path_goal = self.reconstruct_path(came_from_goal, best_meeting_point)
                path_goal.pop(0)
                return path_start + path_goal[::-1]

        return []

    def unidirectional_search(self):
        self.OPEN = []
        self.CLOSED = set()
        self.PARENT = dict()
        self.g = dict()
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))
        count = 0

        while self.OPEN:
            count += 1
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.add(s)
            if s == self.s_goal:
                break
            for s_n in self.get_neighbor(s):
                if s_n in self.CLOSED:
                    continue
                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        path = self.extract_path(self.PARENT)
        visited = list(self.CLOSED)
        return path, visited, count, self.g

    def get_neighbor(self, s):
        neighbors = [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]
        return [n for n in neighbors if self.is_valid(n)]

    def is_valid(self, s):
        if not (0 <= s[0] < self.range_x[1] and 0 <= s[1] < self.range_y[1]):
            return False
        for horizontal in self.horizontal_barriers:
            if horizontal[0] <= s[1] <= horizontal[1] and s[0] == horizontal[2]:
                return False
        for vertical in self.vertical_barriers:
            if vertical[0] <= s[0] <= vertical[1] and s[1] == vertical[2]:
                return False
        return True

    def cost(self, s_start, s_goal):
        if self.is_collision(s_start, s_goal):
            return math.inf
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        line1 = [s_start, s_end]
        for horizontal in self.horizontal_barriers:
            line2 = [[horizontal[1], horizontal[0]], [horizontal[2], horizontal[0]]]
            if is_lines_collision(line1, line2):
                return True
        for vertical in self.vertical_barriers:
            line2 = [[vertical[0], vertical[1]], [vertical[0], vertical[2]]]
            if is_lines_collision(line1, line2):
                return True
        for x in self.range_x:
            line2 = [[x, self.range_y[0]], [x, self.range_y[1]]]
            if is_lines_collision(line1, line2):
                return True
        for y in self.range_y:
            line2 = [[self.range_x[0], y], [self.range_x[1], y]]
            if is_lines_collision(line1, line2):
                return True
        return False

    def f_value(self, s):
        return self.g[s] + self.heuristic(s)

    def heuristic(self, s):
        return self._euclidean_distance_squared(s, self.s_goal)

    def _euclidean_distance_squared(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def reconstruct_path(self, came_from, node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    def extract_path(self, PARENT):
        path = [self.s_goal]
        s = self.s_goal
        while s != self.s_start:
            s = PARENT[s]
            path.append(s)
        return list(reversed(path))
