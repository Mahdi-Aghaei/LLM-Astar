import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from .env import *  # فرض بر این است که این ماژول، تعریف‌هایی دارد که به درستی در پروژه شما موجود است.

class Plotting:
    def __init__(self, xI, xG, env):
        self.xI, self.xG = xI, xG
        self.env = env
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        self.obs = obs
    
    def plot_map(self, name, path="temp.png", show=False):
        plt.clf()
        self.plot_grid(name)
        plt.savefig(path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()

    def animation(self, path, visited, show, name, filepath):
        plt.clf()
        self.plot_grid(name)
        self.plot_visited(visited)
        if path:
            self.plot_path(path)
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        # تغییر رنگ نقاط شروع و هدف
        plt.plot(self.xI[0], self.xI[1], "o", color='deepskyblue', markersize=10, label="Start")
        plt.plot(self.xG[0], self.xG[1], "o", color='forestgreen', markersize=10, label="Goal")
        
        # تغییر رنگ موانع
        plt.scatter(obs_x, obs_y, color='gray', alpha=0.6, s=50, label="Obstacles", edgecolors='black', linewidths=0.5)

        # زیباسازی شبکه (grid) و نمایش محدوده
        plt.title(name, fontsize=16, fontweight='bold', color='darkslategray')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
        plt.legend(loc='upper left', fontsize=10)
        plt.axis('equal')

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0
        # استفاده از گرادیان رنگی برای نشان دادن نقاط بازدید شده
        colors = cm.viridis(np.linspace(0, 1, len(visited)))

        for i, x in enumerate(visited):
            plt.plot(x[0], x[1], color=colors[i], marker='o', markersize=6)
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            count += 1

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        # استفاده از رنگ‌های متنوع برای نمایش مسیر
        if not flag:
            plt.plot(path_x, path_y, linewidth=4, color=cl, marker='o', markersize=5)
        else:
            plt.plot(path_x, path_y, linewidth=4, color=cl, marker='o', markersize=5)

        plt.plot(self.xI[0], self.xI[1], "o", color='deepskyblue', markersize=10)
        plt.plot(self.xG[0], self.xG[1], "o", color='forestgreen', markersize=10)

        plt.pause(0.01)

    def color_list(self):
        cl_v = ['silver', 'wheat', 'lightskyblue', 'royalblue', 'slategray']
        cl_p = ['gray', 'orange', 'deepskyblue', 'red', 'm']
        return cl_v, cl_p

    def color_list_2(self):
        cl = ['silver', 'steelblue', 'dimgray', 'cornflowerblue', 'dodgerblue', 'royalblue', 'plum', 'mediumslateblue', 'mediumpurple', 'blueviolet']
        return cl
