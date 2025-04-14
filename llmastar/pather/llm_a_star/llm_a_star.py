import json
import math
import heapq
import threading  # برای پردازش موازی

from llmastar.env.search import env, plotting
from llmastar.model import ChatGPT, Llama3
from llmastar.utils import is_lines_collision, list_parse
from .prompt import *

class LLMAStar:
    """بهینه‌سازی الگوریتم LLM-A* برای سرعت بیشتر"""

    GPT_METHOD = "PARSE"
    GPT_LLMASTAR_METHOD = "LLM-A*"

    def __init__(self, llm='gpt', prompt='standard'):
        self.llm = llm
        if self.llm == 'gpt':
            self.parser = ChatGPT(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = ChatGPT(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        elif self.llm == 'llama':
            self.model = Llama3()
        else:
            raise ValueError("مدل LLM معتبر نیست. 'gpt' یا 'llama' را انتخاب کنید.")
        
        assert prompt in ['standard', 'cot', 'repe'], "نوع پرس و جو معتبر نیست. 'standard', 'cot', یا 'repe' را انتخاب کنید."
        self.prompt = prompt

    def _parse_query(self, query):
        """پارس کردن ورودی با استفاده از مدل LLM مشخص شده"""
        if isinstance(query, str):
            if self.llm == 'gpt':
                response = self.parser.chat(query)
                return json.loads(response)
            elif self.llm == 'llama':
                response = self.model.ask(parse_llama.format(query=query))
                return json.loads(response)
        return query

    def _initialize_parameters(self, input_data):
        """مقداردهی اولیه به پارامترهای محیط از داده‌های ورودی"""
        self.s_start = tuple(input_data['start'])
        self.s_goal = tuple(input_data['goal'])
        self.horizontal_barriers = input_data['horizontal_barriers']
        self.vertical_barriers = input_data['vertical_barriers']
        self.range_x = input_data['range_x']
        self.range_y = input_data['range_y']
        self.Env = env.Env(self.range_x[1], self.range_y[1], self.horizontal_barriers, self.vertical_barriers)
        self.plot = plotting.Plotting(self.s_start, self.s_goal, self.Env)
        self.range_x[1] -= 1
        self.range_y[1] -= 1
        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.OPEN = []
        self.CLOSED = set()
        self.PARENT = dict()
        self.g = dict()

    def _initialize_llm_paths(self):
        """مقداردهی اولیه مسیرها با استفاده از پیشنهادات LLM"""
        start, goal = list(self.s_start), list(self.s_goal)
        query = self._generate_llm_query(start, goal)

        if self.llm == 'gpt':
            response = self.model.ask(prompt=query, max_tokens=1000)
        elif self.llm == 'llama':
            response = self.model.ask(prompt=query)
        nodes = list_parse(response)
        self.target_list = self._filter_valid_nodes(nodes)

        if not self.target_list or self.target_list[0] != self.s_start:
            self.target_list.insert(0, self.s_start)
        if not self.target_list or self.target_list[-1] != self.s_goal:
            self.target_list.append(self.s_goal)

        self.i = 1
        self.s_target = self.target_list[1]

    def _generate_llm_query(self, start, goal):
        """ساخت پرس و جو برای LLM"""
        if self.llm == 'gpt':
            return gpt_prompt[self.prompt].format(start=start, goal=goal,
                                horizontal_barriers=self.horizontal_barriers,
                                vertical_barriers=self.vertical_barriers)
        elif self.llm == 'llama':
            return llama_prompt[self.prompt].format(start=start, goal=goal,
                                    horizontal_barriers=self.horizontal_barriers,
                                    vertical_barriers=self.vertical_barriers)

    def _filter_valid_nodes(self, nodes):
        """فیلتر کردن گره‌های نامعتبر براساس محدودیت‌های محیطی"""
        return [(node[0], node[1]) for node in nodes
                if (node[0], node[1]) not in self.obs
                and self.range_x[0] + 1 < node[0] < self.range_x[1] - 1
                and self.range_y[0] + 1 < node[1] < self.range_y[1] - 1]

    def searching(self, query, filepath='temp.png'):
        """الگوریتم A* جستجو با بهینه‌سازی‌های مختلف"""
        self.filepath = filepath
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.add(s)

            if s == self.s_goal:
                break

            # استفاده از چند رشته‌ای برای بهینه‌سازی انتخاب همسایگان
            threads = []
            for s_n in self.get_neighbor(s):
                if s_n in self.CLOSED:
                    continue

                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    thread = threading.Thread(target=self._update_queue, args=(s_n,))
                    threads.append(thread)
                    thread.start()

            for thread in threads:
                thread.join()

        path = self.extract_path(self.PARENT)
        visited = list(self.CLOSED)
        result = {
            "operation": len(self.CLOSED),
            "storage": len(self.g),
            "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
            "llm_output": self.target_list
        }
        self.plot.animation(path, visited, True, "LLM-A*", self.filepath)
        return result

    @staticmethod
    def _euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _update_queue(self, s_n):
        """به روز رسانی لیست باز به صورت بهینه"""
        heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

    def get_neighbor(self, s):
        """یافتن همسایگان گره s"""
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """محاسبه هزینه حرکت از s_start به s_goal"""
        return math.inf if self.is_collision(s_start, s_goal) else math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """بررسی برخورد خط (s_start, s_end) با موانع"""
        line1 = [s_start, s_end]
        return any(is_lines_collision(line1, [[h[1], h[0]], [h[2], h[0]]]) for h in self.horizontal_barriers) or \
               any(is_lines_collision(line1, [[v[0], v[1]], [v[0], v[2]]]) for v in self.vertical_barriers) or \
               any(is_lines_collision(line1, [[x, self.range_y[0]], [x, self.range_y[1]]]) for x in self.range_x) or \
               any(is_lines_collision(line1, [[self.range_x[0], y], [self.range_x[1], y]]) for y in self.range_y)

    def f_value(self, s):
        """محاسبه f-value برای گره s"""
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """استخراج مسیر بر اساس والدین"""
        path = [self.s_goal]
        while path[-1] != self.s_start:
            path.append(PARENT[path[-1]])
        return path[::-1]

    def heuristic(self, s):
        """محاسبه مقدار هورسیتیک برای گره s"""
        return math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])

