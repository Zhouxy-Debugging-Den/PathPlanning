"""
A_star 2D
@author: huiming zhou
带中文逐行注释版本
"""

import os
import sys
import math
import heapq

# 将项目的上级路径加入系统路径，方便导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

# 导入绘图工具和环境
from Search_2D import plotting, env


class AStar:
    """A* 算法类，优先级由 f = g + h 决定
    g: 从起点到当前点的实际代价
    h: 从当前点到目标点的启发式估计代价
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        # 起点和目标点
        self.s_start = s_start
        self.s_goal = s_goal
        # 启发函数类型（曼哈顿 / 欧几里得）
        self.heuristic_type = heuristic_type

        # 环境对象，包含地图、障碍物等信息
        self.Env = env.Env()

        # 可行的移动集合（比如上下左右或对角线）
        self.u_set = self.Env.motions
        # 障碍物位置集合
        self.obs = self.Env.obs

        # OPEN表：优先队列（小顶堆），存储待访问节点
        self.OPEN = []
        # CLOSED表：已访问节点（扩展顺序）
        self.CLOSED = []
        # PARENT字典：存储节点的父节点，用于回溯路径
        self.PARENT = dict()
        # g字典：存储每个节点的代价值（从起点到该点的代价）
        self.g = dict()

    def searching(self):
        """
        A* 搜索主函数
        :return: 路径和访问顺序
        """
        # 起点的父节点设为自己
        self.PARENT[self.s_start] = self.s_start
        # 起点的代价设为 0
        self.g[self.s_start] = 0
        # 目标点代价初始设为无穷大
        self.g[self.s_goal] = math.inf
        # 将起点加入 OPEN 表，优先级由 f 值决定
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        # 当 OPEN 表不为空时循环
        while self.OPEN:
            # 弹出 f 值最小的节点
            _, s = heapq.heappop(self.OPEN)
            # 将该节点加入 CLOSED 表
            self.CLOSED.append(s)

            # 如果到达目标点，搜索结束
            if s == self.s_goal:
                break

            # 遍历邻居节点
            for s_n in self.get_neighbor(s):
                # 计算新代价
                new_cost = self.g[s] + self.cost(s, s_n)

                # 如果邻居还未在 g 中记录，初始化为无穷大
                if s_n not in self.g:
                    self.g[s_n] = math.inf

                # 如果找到更优路径，则更新 g 值和父节点
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    # 将邻居加入 OPEN 表，按 f 值排序
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        # 返回路径和访问顺序
        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        重复 A*（ARA* 的思想）
        :param e: 初始权重（A* 的变形，f = g + e*h）
        :return: 各阶段路径和访问顺序
        """
        path, visited = [], []

        # 不断减小 e 的值（比如从 2.5 -> 2.0 -> 1.5 -> 1.0）
        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        带权 A* 的一次搜索
        :param s_start: 起点
        :param s_goal: 目标点
        :param e: 权重（f = g + e*h）
        :return: 路径和访问顺序
        """
        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        # 起点入队，优先级 f = g + e*h
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        获取状态 s 的邻居节点
        :param s: 当前状态
        :return: 邻居节点列表
        """
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        计算移动代价
        :param s_start: 起点
        :param s_goal: 终点
        :return: 代价（若碰撞返回无穷大）
        """
        if self.is_collision(s_start, s_goal):
            return math.inf

        # 欧几里得距离
        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        判断线段 (s_start, s_end) 是否与障碍物碰撞
        :return: True 碰撞 / False 不碰撞
        """
        # 起点或终点是障碍物
        if s_start in self.obs or s_end in self.obs:
            return True

        # 对角线移动时的障碍检测
        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            #  判断是哪种对角线（↘ ↙ ↗ ↖）
            # 对角线方向1，取到两端点的 "中间格子"
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            # 对角线方向2，取到另一种对角线的 "中间格子"
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        计算 f = g + h
        :param s: 当前状态
        :return: f 值
        """
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        根据父节点字典回溯路径
        :return: 从起点到终点的路径
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)
            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        启发函数 h(s)
        :param s: 当前状态
        :return: 启发值
        """
        heuristic_type = self.heuristic_type
        goal = self.s_goal

        if heuristic_type == "manhattan":
            # 曼哈顿距离
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            # 默认用欧几里得距离
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    # 定义起点和终点
    s_start = (5, 5)
    s_goal = (45, 25)

    # 初始化 A*，启发式函数使用欧几里得
    astar = AStar(s_start, s_goal, "euclidean")
    plot = plotting.Plotting(s_start, s_goal)

    # 执行 A* 搜索
    path, visited = astar.searching()
    plot.animation(path, visited, "A*")  # 动画演示

    # 测试重复 A*（ARA*）
    # path, visited = astar.searching_repeated_astar(2.5)
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
