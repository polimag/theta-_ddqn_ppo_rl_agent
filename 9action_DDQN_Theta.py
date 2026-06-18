"""
DDQN

Режимы запуска:
  train_test = 'train'  — обучение
  train_test = 'test1'  — тестирование на рандомных waypoints (без препятствий)
  train_test = 'test2'  — тестирование с Theta* + статические препятствия
"""

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import heapq
from collections import namedtuple, deque
from typing import List
import os
import time

# =====================================================================
# ПАРАМЕТРЫ
# =====================================================================
H, W = 200, 400             # Размер дискретного поля

n_ep = 200                  # Число эпизодов обучения
n_prn = 20                  # Частота вывода статистики в консоль
t_max = int(2.5 * (H + W))+ 500  # Максимум шагов в эпизоде

in_dim = 3                  # Вход сети: нормированный вектор до цели + cte
action_space_dim = 9        # 8 направлений + «стоим на месте»

WP_THRESHOLD = 2.0          # Порог достижения waypoint, пикселей

vis_every_n_successes = 20

train_test = 'test1'        # train, test1, test2
model_path = 'best_model_dqn.pth'

use_seed = False
if use_seed:
    s = 42
    np.random.seed(s); torch.manual_seed(s); random.seed(s)

layers_sizes = (in_dim, 64, 32, action_space_dim)

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def get_cte(pos: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> float:
    """CTE: расстояние от позиции агента до отрезка [p_start, p_end]"""
    line_vec = p_end - p_start
    point_vec = pos   - p_start
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq < 1e-9:
        return float(np.linalg.norm(pos - p_start))
    t = float(np.clip(np.dot(point_vec, line_vec) / line_len_sq, 0.0, 1.0))
    closest = p_start + t * line_vec
    return float(np.linalg.norm(pos - closest))


def norm_dir(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Нормированный вектор от pos до target — вход агента"""
    v = target - pos
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)


def make_action_vectors(stp: float) -> np.ndarray:
    d = stp / math.sqrt(2)
    return np.array([
        [ stp,  0.  ],   # 0  вниз
        [-stp,  0.  ],   # 1  вверх
        [ 0.,   stp ],   # 2  вправо
        [ 0.,  -stp ],   # 3  влево
        [ d,    d   ],   # 4  вниз-вправо
        [-d,    d   ],   # 5  вверх-вправо
        [ d,   -d   ],   # 6  вниз-влево
        [-d,   -d   ],   # 7  вверх-влево
        [ 0.,   0.  ],   # 8  стоять на месте
    ], dtype=np.float32)


def make_state(pos: np.ndarray,
               seg_start: np.ndarray,
               seg_end: np.ndarray) -> np.ndarray:
    direction = norm_dir(pos, seg_end)

    seg_vec = seg_end - seg_start
    pos_vec = pos - seg_start
    cross = float(seg_vec[0] * pos_vec[1] - seg_vec[1] * pos_vec[0])
    cte_abs = get_cte(pos, seg_start, seg_end)
    cte_signed = cte_abs * (1.0 if cross >= 0 else -1.0)
    cte_norm = float(np.clip(cte_signed / 30.0, -1.0, 1.0))

    return np.array([direction[0], direction[1], cte_norm], dtype=np.float32)


# =====================================================================
# ПРЕПЯТСТВИЯ
# =====================================================================

class Obstacle:
    def __init__(self, y, x, h, w):
        self.y, self.x, self.h, self.w = y, x, h, w

    def contains(self, y, x):
        return self.y <= y < self.y + self.h and self.x <= x < self.x + self.w

    def expanded(self, m=3):
        return Obstacle(max(0, self.y - m), max(0, self.x - m),
                        min(H, self.h + 2 * m), min(W, self.w + 2 * m))


# =====================================================================
# Theta*
# =====================================================================

class ThetaStarPlanner:

    def __init__(self, obstacles: List[Obstacle], gs: int = 5):
        self.obs = obstacles
        self.gs = gs

    def valid(self, y, x):
        if not (0 <= y < H and 0 <= x < W):
            return False
        return all(not o.expanded(3).contains(y, x) for o in self.obs)

    def h(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def neighbors(self, node):
        y, x = node
        nb = []
        for dy in [-self.gs, 0, self.gs]:
            for dx in [-self.gs, 0, self.gs]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if self.valid(ny, nx):
                    nb.append(((ny, nx), math.hypot(dy, dx)))
        return nb

    def line_of_sight(self, p1, p2):
        """Алгоритм Брезенхэма"""
        y1, x1 = int(p1[0]), int(p1[1])
        y2, x2 = int(p2[0]), int(p2[1])
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        sy = 1 if y1 < y2 else -1
        sx = 1 if x1 < x2 else -1
        e  = dx - dy
        while True:
            if not self.valid(y1, x1):
                return False
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * e
            if e2 > -dy:
                e  -= dy
                x1 += sx
            if e2 < dx:
                e  += dx
                y1 += sy
        return True

    def plan(self, start, goal):
        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]),  int(goal[1]))

        open_set = [(0.0, s)]
        came_from = {s: s}
        g_score = {s: 0.0}

        while open_set:
            _, cur = heapq.heappop(open_set)

            if self.h(cur, g) < self.gs * 1.5:
                path = []
                node = cur
                while node != came_from[node]:
                    path.append(np.array(node, dtype=np.float32))
                    node = came_from[node]
                path.append(np.array(s, dtype=np.float32))
                path.reverse()
                if self.h(tuple(path[-1]), g) > self.gs:
                    path.append(np.array(g, dtype=np.float32))
                return path

            for nb, step_cost in self.neighbors(cur):
                parent = came_from[cur]
                if self.line_of_sight(np.array(parent), np.array(nb)):
                    tentative_g = g_score[parent] + self.h(parent, nb)
                    if tentative_g < g_score.get(nb, float('inf')):
                        came_from[nb] = parent
                        g_score[nb] = tentative_g
                        heapq.heappush(open_set,
                                       (tentative_g + self.h(nb, g), nb))
                else:
                    tentative_g = g_score[cur] + step_cost
                    if tentative_g < g_score.get(nb, float('inf')):
                        came_from[nb] = cur
                        g_score[nb] = tentative_g
                        heapq.heappush(open_set,
                                       (tentative_g + self.h(nb, g), nb))

        return [np.array(s, dtype=np.float32), np.array(g, dtype=np.float32)]
    
    def simplify_path(self, path):
        simplified = list(path)
        i = 0
        while i < len(simplified) - 2:
            if self.line_of_sight(simplified[i], simplified[i + 2]):
                simplified.pop(i + 1)
            else:
                i += 1
        return simplified


# =====================================================================
# СРЕДА ОБУЧЕНИЯ
# =====================================================================

class TrainEnv:

    CTE_INIT_MIN = 0.0

    def __init__(self):
        self.stp = 1.0
        self.av = make_action_vectors(self.stp)
        self.ep_count = 0
        self.seg_start = None
        self.seg_end = None
        self.pos = None
        self.cte_hist  = []

    def _new_goal(self) -> np.ndarray:
        margin = 15
        side = np.random.randint(4)
        if side == 0:
            return np.array([margin,np.random.uniform(margin, W-margin)], dtype=np.float32)
        elif side == 1:
            return np.array([H-margin, np.random.uniform(margin, W-margin)], dtype=np.float32)
        elif side == 2:
            return np.array([np.random.uniform(margin, H-margin), margin], dtype=np.float32)
        else:
            return np.array([np.random.uniform(margin, H-margin), W-margin], dtype=np.float32)

    def _start_pos(self) -> np.ndarray:
        return self.seg_start.copy()

    def reset(self) -> np.ndarray:
        self.ep_count += 1
        self.seg_start = np.array([H/2.0, W/2.0], dtype=np.float32)
        self.seg_end = self._new_goal()
        self.pos = self._start_pos()
        self.cte_hist = []
        self.prev_cte = 0.0
        return make_state(self.pos, self.seg_start, self.seg_end)

    def step(self, action: int):
        prev = self.pos.copy()
        self.pos = self.pos + self.av[action]

        if not (0 <= self.pos[0] < H and 0 <= self.pos[1] < W):
            self.pos = prev
            return make_state(self.pos, self.seg_start, self.seg_end), -5.0, True, False

        cte = get_cte(self.pos, self.seg_start, self.seg_end)
        self.cte_hist.append(cte)

        to_goal = self.seg_end - self.pos
        dist_to_goal = float(np.linalg.norm(to_goal))

        if dist_to_goal > 1e-6:
            goal_dir = to_goal / dist_to_goal
            progress = float(np.dot(self.av[action], goal_dir))
        else:
            progress = 0.0

        cte_ratio = cte / WP_THRESHOLD
        cte_pen = 0.3 * cte_ratio + 0.1 * cte_ratio ** 2

        delta_cte = cte - self.prev_cte
        delta_pen = delta_cte * 3.0
        self.prev_cte = cte

        reward = progress - cte_pen - delta_pen

        if action == 8:
            reward -= 10.0

        if dist_to_goal < WP_THRESHOLD:
            return make_state(self.pos, self.seg_start, self.seg_end), 100.0, True, True

        return make_state(self.pos, self.seg_start, self.seg_end), reward, False, False

    def mean_cte(self) -> float:
        return float(np.mean(self.cte_hist)) if self.cte_hist else 0.0


# =====================================================================
# СРЕДА ДЛЯ test1
# =====================================================================

class TestEnv:

    def __init__(self, n_waypoints: int = None):
        self.stp = 1.0
        self.av = make_action_vectors(self.stp)
        n_wp = n_waypoints if n_waypoints is not None else np.random.randint(2, 6)
        self.wps = self._gen(n_wp)
        self.wi = 1
        self.pos = self.wps[0].copy()
        self.path_taken = [self.pos.copy()]
        self.cte_hist = []

    def _gen(self, n_wp: int) -> List[np.ndarray]:
        mg = 20
        pts = []
        side = np.random.randint(2)
        if side == 0:
            pts.append(np.array([np.random.uniform(mg, H-mg), mg + np.random.uniform(0, 15)], dtype=np.float32))
        else:
            pts.append(np.array([mg + np.random.uniform(0, 15), np.random.uniform(mg, W-mg)], dtype=np.float32))
        for _ in range(n_wp - 1):
            pts.append(np.array([np.random.uniform(mg, H-mg), np.random.uniform(mg, W-mg)], dtype=np.float32))
        if side == 0:
            pts.append(np.array([np.random.uniform(mg, H-mg), W - mg - np.random.uniform(0, 15)], dtype=np.float32))
        else:
            pts.append(np.array([H - mg - np.random.uniform(0, 15), np.random.uniform(mg, W-mg)], dtype=np.float32))
        return pts

    def reset(self) -> np.ndarray:
        self.wi = 1
        self.pos = self.wps[0].copy()
        self.path_taken = [self.pos.copy()]
        self.cte_hist = []
        self.prev_cte = 0.0
        return make_state(self.pos, self.wps[0], self.wps[1])

    def step(self, action: int):
        prev = self.pos.copy()
        self.pos = self.pos + self.av[action]

        seg_s = self.wps[self.wi - 1]
        seg_e = self.wps[self.wi]

        if not (0 <= self.pos[0] < H and 0 <= self.pos[1] < W):
            self.pos = prev
            return make_state(self.pos, seg_s, seg_e), -5.0, True, False

        self.path_taken.append(self.pos.copy())

        cte = get_cte(self.pos, seg_s, seg_e)
        self.cte_hist.append(cte)

        to_wp = seg_e - self.pos
        dist_to_wp = float(np.linalg.norm(to_wp))

        if dist_to_wp > 1e-6:
            goal_dir = to_wp / dist_to_wp
            progress = float(np.dot(self.av[action], goal_dir))
        else:
            progress = 0.0

        cte_ratio = cte / WP_THRESHOLD
        cte_pen = 0.3 * cte_ratio + 0.1 * cte_ratio ** 2

        delta_cte = cte - self.prev_cte
        delta_pen = delta_cte * 3.0
        self.prev_cte = cte

        reward = progress - cte_pen - delta_pen

        if action == 8:
            reward -= 1.0

        if dist_to_wp < WP_THRESHOLD:
            if self.wi < len(self.wps) - 1:
                self.wi += 1
                self.prev_cte = 0.0
                return make_state(self.pos, self.wps[self.wi-1], self.wps[self.wi]), 10.0, False, False
            else:
                return make_state(self.pos, seg_s, seg_e), 100.0, True, True

        return make_state(self.pos, seg_s, seg_e), reward, False, False

    def mean_cte(self) -> float:
        return float(np.mean(self.cte_hist)) if self.cte_hist else 0.0

    def visualize(self, title: str = "", ax=None, show: bool = True,
                  reverse_path: list = None):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(14, 7))

        ax.set_xlim(-1, W+1); ax.set_ylim(H+1, -1)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')

        wp = np.array(self.wps)
        ax.plot(wp[:,1], wp[:,0], 'b--', lw=2, label='Плановый путь', zorder=3)

        for i, p in enumerate(self.wps[1:-1], 1):
            ax.plot(p[1], p[0], 'bo', ms=8, zorder=4)
            ax.annotate(f'WP{i}', (p[1], p[0]),
                        textcoords='offset points', xytext=(5,5), fontsize=9)

        if len(self.path_taken) > 1:
            pt = np.array(self.path_taken)
            ax.plot(pt[:,1], pt[:,0], color='#e06000', lw=1.8,
                    label='Путь агента (прямой)', zorder=3)

        if reverse_path is not None and len(reverse_path) > 1:
            rp = np.array(reverse_path)
            ax.plot(rp[:,1], rp[:,0], color='#22aa22', lw=1.8,
                    label='Путь агента (обратный)', zorder=3)

        ax.plot(self.wps[0][1],  self.wps[0][0],  'g^', ms=12,
                label='Старт', zorder=5)
        ax.plot(self.wps[-1][1], self.wps[-1][0], 'rs', ms=12,
                label='Финиш', zorder=5)

        ax.set_title(
            f'{title}\n'
            f'Среднее CTE: {self.mean_cte():.2f} px  |  '
            f'WP: {len(self.wps)-2}  |  Шагов: {len(self.path_taken)}',
            fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        if standalone:
            plt.tight_layout()
            if show:
                plt.show()
            return fig


# =====================================================================
# СРЕДА ДЛЯ test2
# =====================================================================

class ThetaStarTestEnv:

    N_OBSTACLES_MIN = 10
    N_OBSTACLES_MAX = 15

    def __init__(self):
        self.stp = 1.0
        self.av = make_action_vectors(self.stp)
        self.obstacles: List[Obstacle] = []
        self.wps: List[np.ndarray] = []
        self.wi = 1
        self.pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.path_taken: List[np.ndarray] = []
        self.cte_hist: List[float] = []
        self.prev_cte = 0.0
        self._generate()

    def _generate(self):
        margin = 20
        n_obs = np.random.randint(self.N_OBSTACLES_MIN, self.N_OBSTACLES_MAX + 1)

        # Старт и финиш у противоположных краёв поля
        side = np.random.randint(2)
        if side == 0:
            start = np.array([np.random.uniform(margin, H - margin), margin + np.random.uniform(0, 15)], dtype=np.float32)
            goal = np.array([np.random.uniform(margin, H - margin), W - margin - np.random.uniform(0, 15)], dtype=np.float32)
        else:
          start = np.array([np.random.uniform(margin, H - margin), W - margin - np.random.uniform(0, 15)], dtype=np.float32)
          goal = np.array([np.random.uniform(margin, H - margin), margin + np.random.uniform(0, 15)], dtype=np.float32)

        # Генерация препятствий
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < n_obs and attempts < 300:
            attempts += 1
            oh = np.random.randint(15, 50)
            ow = np.random.randint(15, 70)
            oy = np.random.randint(margin, H - margin - oh)
            ox = np.random.randint(margin, W - margin - ow)
            obs = Obstacle(oy, ox, oh, ow)
            clear_start = not obs.expanded(12).contains(int(start[0]), int(start[1]))
            clear_goal = not obs.expanded(12).contains(int(goal[0]),  int(goal[1]))
            no_overlap = not any(
                obs.expanded(5).y < (o.y + o.h) and
                (obs.expanded(5).y + obs.expanded(5).h) > o.y and
                obs.expanded(5).x < (o.x + o.w) and
                (obs.expanded(5).x + obs.expanded(5).w) > o.x
                for o in self.obstacles
            )
            if clear_start and clear_goal and no_overlap:
                self.obstacles.append(obs)

        # Theta*
        planner = ThetaStarPlanner(self.obstacles, gs=5)
        self.wps = planner.plan(start, goal)

        self.wi = 1
        self.pos = self.wps[0].copy()
        self.path_taken = [self.pos.copy()]
        self.cte_hist = []
        self.prev_cte = 0.0

    def reset(self) -> np.ndarray:
        self._generate()
        return make_state(self.pos, self.wps[0], self.wps[1])

    def hits_obstacle(self, pos: np.ndarray) -> bool:
        y, x = int(pos[0]), int(pos[1])
        return any(o.contains(y, x) for o in self.obstacles)

    def step(self, action: int):
        prev = self.pos.copy()
        self.pos = self.pos + self.av[action]

        seg_s = self.wps[self.wi - 1]
        seg_e = self.wps[self.wi]

        # Выход за границы
        if not (0 <= self.pos[0] < H and 0 <= self.pos[1] < W):
            self.pos = prev
            return make_state(self.pos, seg_s, seg_e), -5.0, True, False

        # Столкновение с препятствием
        if self.hits_obstacle(self.pos):
            self.pos = prev
            return make_state(self.pos, seg_s, seg_e), -5.0, True, False

        self.path_taken.append(self.pos.copy())

        cte = get_cte(self.pos, seg_s, seg_e)
        self.cte_hist.append(cte)

        to_wp = seg_e - self.pos
        dist_to_wp = float(np.linalg.norm(to_wp))

        if dist_to_wp > 1e-6:
            goal_dir = to_wp / dist_to_wp
            progress = float(np.dot(self.av[action], goal_dir))
        else:
            progress = 0.0

        cte_ratio = cte / WP_THRESHOLD
        cte_pen = 0.3 * cte_ratio + 0.1 * cte_ratio ** 2

        delta_cte = cte - self.prev_cte
        delta_pen = delta_cte * 3.0
        self.prev_cte = cte

        reward = progress - cte_pen - delta_pen

        if action == 8:
            reward -= 1.0

        if dist_to_wp < WP_THRESHOLD:
            if self.wi < len(self.wps) - 1:
                self.wi += 1
                self.prev_cte = 0.0
                return make_state(self.pos, self.wps[self.wi-1],
                                  self.wps[self.wi]), 10.0, False, False
            else:
                return make_state(self.pos, seg_s, seg_e), 100.0, True, True

        return make_state(self.pos, seg_s, seg_e), reward, False, False

    def mean_cte(self) -> float:
        return float(np.mean(self.cte_hist)) if self.cte_hist else 0.0

    def visualize(self, title: str = "", show: bool = True):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(-1, W + 1); ax.set_ylim(H + 1, -1)
        ax.set_aspect('equal'); ax.set_facecolor('#f5f5f5')

        # Препятствия — серые прямоугольники
        for obs in self.obstacles:
            rect = plt.Rectangle((obs.x, obs.y), obs.w, obs.h,
                                  linewidth=1, edgecolor='#444',
                                  facecolor='#aaaaaa', alpha=0.75, zorder=2)
            ax.add_patch(rect)

        # Путь Theta* - синий пунктир
        wp = np.array(self.wps)
        ax.plot(wp[:, 1], wp[:, 0], 'b--', lw=2, label='Путь Theta*', zorder=3)
        for i, p in enumerate(self.wps[1:-1], 1):
            ax.plot(p[1], p[0], 'bo', ms=7, zorder=4)
            ax.annotate(f'WP{i}', (p[1], p[0]),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)

        # Путь агента - оранжевый
        if len(self.path_taken) > 1:
            pt = np.array(self.path_taken)
            ax.plot(pt[:, 1], pt[:, 0], color='#e06000', lw=1.8,
                    label='Путь агента', zorder=3)

        ax.plot(self.wps[0][1],  self.wps[0][0],  'g^', ms=12, label='Старт', zorder=5)
        ax.plot(self.wps[-1][1], self.wps[-1][0], 'rs', ms=12, label='Финиш', zorder=5)

        ax.set_title(
            f'{title}\n'
            f'CTE: {self.mean_cte():.2f} px  |  '
            f'WP: {len(self.wps) - 2}  |  '
            f'Шагов: {len(self.path_taken)}  |  '
            f'Препятствий: {len(self.obstacles)}',
            fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig


# =====================================================================
# НЕЙРОННАЯ СЕТЬ
# =====================================================================

class DQN(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(),
            nn.Linear(sz[1], sz[2]), nn.ReLU(),
            nn.Linear(sz[2], sz[3]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(0) if x.dim() == 1 else x)


class ReplayMemory:
    def __init__(self, cap):
        self.mem = deque(maxlen=cap)

    def push(self, *a):
        self.mem.append(Transition(*a))

    def sample(self, n):
        return random.sample(self.mem, n)

    def __len__(self):
        return len(self.mem)



batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.05
eps_decay = 8000
tau = 0.005
lr = 1e-3

steps_done = 0
policy_net = DQN(layers_sizes).to(device)
target_net = DQN(layers_sizes).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
mem = ReplayMemory(50000)


def select_action(st: torch.Tensor, eps: float) -> torch.Tensor:
    global steps_done
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(st).max(1).indices.view(1, 1)
    return torch.tensor([[np.random.randint(action_space_dim)]],
                        device=device, dtype=torch.long)


def optimize():
    if len(mem) < batch_size:
        return
    batch = Transition(*zip(*mem.sample(batch_size)))

    nf  = torch.tensor([s is not None for s in batch.next_state],
                        device=device, dtype=torch.bool)
    nfs = torch.cat([s for s in batch.next_state if s is not None])

    sb = torch.cat(batch.state)
    ab = torch.cat(batch.action)
    rb = torch.cat(batch.reward)

    qv = policy_net(sb).gather(1, ab)

    nq = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        ba = policy_net(nfs).max(1).indices
        nq[nf] = target_net(nfs).gather(1, ba.unsqueeze(1)).squeeze(1)

    loss = F.smooth_l1_loss(qv, (rb + gamma * nq).unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    for pp, tp in zip(policy_net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)


# =====================================================================
# ВИЗУАЛИЗАЦИЯ ВО ВРЕМЯ ОБУЧЕНИЯ
# =====================================================================

def show_training_progress(ep: int, n_solved: int):
    policy_net.eval()

    env = TrainEnv()
    env.ep_count = 9999
    env.seg_start = np.array([H / 2.0, W / 2.0], dtype=np.float32)
    env.seg_end = env._new_goal()
    env.pos = env.seg_start.copy()
    env.cte_hist = []
    state = make_state(env.pos, env.seg_start, env.seg_end)

    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    solved = False
    path = [env.pos.copy()]

    for _ in range(t_max):
        with torch.no_grad():
            a = policy_net(st).max(1).indices.view(1, 1)
        ns, _, done, solved = env.step(a.item())
        path.append(env.pos.copy())
        st = torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            break

    status = 'УСПЕХ' if solved else 'НЕУДАЧА'

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, W + 1)
    ax.set_ylim(H + 1, -1)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    pts = np.array([env.seg_start, env.seg_end])
    ax.plot(pts[:, 1], pts[:, 0], 'b--', lw=2, label='Плановый путь', zorder=3)

    if len(path) > 1:
        pa = np.array(path)
        ax.plot(pa[:, 1], pa[:, 0], color='#e06000', lw=1.8,
                label='Путь агента', zorder=3)

    ax.plot(env.seg_start[1], env.seg_start[0], 'g^', ms=12,
            label='Старт (центр поля)', zorder=5)
    ax.plot(env.seg_end[1],   env.seg_end[0],   'rs', ms=12,
            label='Цель (край поля)',  zorder=5)

    ax.set_title(
        f'Обучение — эпизод {ep}, успехов: {n_solved} — {status}\n'
        f'Среднее CTE: {env.mean_cte():.2f} px  |  Шагов: {len(path)}',
        fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1.0)

    policy_net.train()


# =====================================================================
# ОБУЧЕНИЕ
# =====================================================================

def run_training():
    env = TrainEnv()

    ep_rw = []; ep_dur = []; ep_sc = []; ep_cte = []
    n_solved  = 0
    best_sr   = 0.0
    prev_vis  = 0

    print('=' * 72)
    print('ОБУЧЕНИЕ: две точки, без препятствий')
    print(f'Поле: {H}x{W} | Вход: {in_dim} числа | '
          f'Действий: {action_space_dim} | Эпизодов: {n_ep}')
    print(f'Старт: центр поля  |  Цель: случайная точка у края')
    print(f'Порог WP: {WP_THRESHOLD} px  |  '
          f'Визуализация каждые {vis_every_n_successes} успехов')
    print(f'Устройство: {device}')
    print('=' * 72)
    print(f"{'Эп.':>6} | {'Успехов':>7} | {'%':>6} | "
          f"{'Нагр.':>8} | {'Шаги':>7} | {'CTE':>7} | {'eps':>7}")
    print('-' * 72)

    t0 = time.time()

    for ep in range(n_ep):
        eps = eps_end + (eps_start - eps_end) * math.exp(-steps_done / eps_decay)
        state = env.reset()
        st = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        ep_r = 0.0
        solved = False

        for t in range(t_max):
            a = select_action(st, eps)
            ns, r, done, solved = env.step(a.item())
            ep_r += r

            nt = (None if done else
                  torch.tensor(ns, dtype=torch.float32,
                               device=device).unsqueeze(0))
            mem.push(st, a, nt,
                     torch.tensor([r], device=device, dtype=torch.float32))
            if nt is not None:
                st = nt
            optimize()

            if done:
                ep_dur.append(t+1); ep_rw.append(ep_r)
                ep_sc.append(int(solved)); ep_cte.append(env.mean_cte())
                if solved: n_solved += 1
                break

        # Визуализация каждые vis_every_n_successes успехов
        '''if n_solved > 0 and (n_solved - prev_vis) >= vis_every_n_successes:
            prev_vis = n_solved
            print(f'\n  >>> Визуализация: успехов={n_solved}, эп.={ep}\n')
            show_training_progress(ep, n_solved)'''

        if ep > 0 and ep % n_prn == 0:
            w = min(n_prn, len(ep_rw))
            sr_w = sum(ep_sc[-w:]) / w
            print(f"{ep:>6} | {n_solved:>7} | "
                  f"{n_solved/(ep+1)*100:>5.1f}% | "
                  f"{np.mean(ep_rw[-w:]):>8.1f} | "
                  f"{np.mean(ep_dur[-w:]):>7.1f} | "
                  f"{np.mean(ep_cte[-w:]):>7.2f} | "
                  f"{eps:>7.4f}")

            if len(ep_sc) >= 100:
                sr50 = sum(ep_sc[-50:]) / 50
                if sr50 > best_sr:
                    best_sr = sr50
                    torch.save({'policy_net':   policy_net.state_dict(),
                                'episode':      ep,
                                'success_rate': sr50,
                                'n_solved':     n_solved}, model_path)
                    print(f"  >> Сохранена лучшая модель "
                          f"(эп.{ep}, успешность {sr50*100:.1f}%)")

    torch.save({'policy_net':   policy_net.state_dict(),
                'episode':      ep,
                'n_solved':     n_solved,
                'success_rate': best_sr}, 'final_model_dqn.pth')

    print('=' * 72)
    print(f'Завершено за {(time.time()-t0)/60:.1f} мин.  '
          f'Успешно: {n_solved}/{n_ep} ({n_solved/n_ep*100:.1f}%)')
    print('=' * 72)

    return ep_rw, ep_sc, ep_cte


def plot_curves(rw, sc, cte):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    win = 20

    def sm(d):
        return [np.mean(d[max(0,i-win):i+1]) for i in range(len(d))]

    axes[0].plot(sm(rw),  color='steelblue', lw=1.5)
    axes[0].set_title('Средняя награда'); axes[0].set_xlabel('Эпизод')
    axes[0].grid(True, alpha=0.3)

    sr = [np.mean(sc[max(0,i-win):i+1]) for i in range(len(sc))]
    axes[1].plot(sr, color='seagreen', lw=1.5)
    axes[1].axhline(0.95, color='red', ls='--', lw=1, label='95%')
    axes[1].set_title('Успешность'); axes[1].set_xlabel('Эпизод')
    axes[1].set_ylim(0, 1.05); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sm(cte), color='tomato', lw=1.5)
    axes[2].set_title('Среднее CTE'); axes[2].set_xlabel('Эпизод')
    axes[2].set_ylabel('пикс.'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=120, bbox_inches='tight')
    plt.show()
    print('Кривые обучения сохранены: training_curves.png')


# =====================================================================
# ТЕСТИРОВАНИЕ — test1:
# =====================================================================

def _run_episode_dqn(net, env) -> tuple:
    s  = env.reset()
    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
    solved = False
    for _ in range(t_max):
        with torch.no_grad():
            a = net(st).max(1).indices.view(1, 1)
        ns, _, done, solved = env.step(a.item())
        st = torch.tensor(ns, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            break
    return solved, list(env.path_taken), env.mean_cte()


def _visualize_pair_dqn(env_fwd, fwd_path, fwd_cte, fwd_solved,
                        env_rev, rev_path, rev_cte, rev_solved,
                        test_num: int, n_wp: int):
    """Рисует два графика рядом: слева прямой прогон, справа обратный."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    def _draw(ax, env, path, cte, solved, color, direction):
        ax.set_xlim(-1, W+1); ax.set_ylim(H+1, -1)
        ax.set_aspect('equal'); ax.set_facecolor('#f5f5f5')
        wp = np.array(env.wps)
        ax.plot(wp[:,1], wp[:,0], 'b--', lw=2, label='Плановый путь', zorder=3)
        for j, p in enumerate(env.wps[1:-1], 1):
            ax.plot(p[1], p[0], 'bo', ms=8, zorder=4)
            ax.annotate(f'WP{j}', (p[1], p[0]),
                        textcoords='offset points', xytext=(5,5), fontsize=9)
        if len(path) > 1:
            pt = np.array(path)
            ax.plot(pt[:,1], pt[:,0], color=color, lw=1.8,
                    label='Путь агента', zorder=3)
        ax.plot(env.wps[0][1],  env.wps[0][0],  'g^', ms=12, label='Старт', zorder=5)
        ax.plot(env.wps[-1][1], env.wps[-1][0], 'rs', ms=12, label='Финиш', zorder=5)
        status = 'УСПЕХ' if solved else 'НЕУДАЧА'
        ax.set_title(
            f'Тест {test_num} [{direction}]: {status} | WP: {n_wp}\n'
            f'Среднее CTE: {cte:.2f} пкс  |  Шагов: {len(path)}',
            fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    _draw(axes[0], env_fwd, fwd_path, fwd_cte, fwd_solved,
          '#e06000', 'прямой: старт → финиш')
    _draw(axes[1], env_rev, rev_path, rev_cte, rev_solved,
          '#22aa22', 'обратный: финиш → старт')

    plt.tight_layout()
    plt.show()


def run_testing(mpath: str, num_tests: int = 100):
    print(f'\n{"="*60}')
    print(f'ТЕСТИРОВАНИЕ')
    print(f'{"="*60}')
    if not os.path.exists(mpath):
        print(f'Файл не найден: {mpath}')
        return
    ckpt = torch.load(mpath, map_location=device, weights_only=False)
    net = DQN(layers_sizes).to(device)
    net.load_state_dict(ckpt['policy_net'])
    net.eval()
    print(f"Эпизод: {ckpt.get('episode','?')}  |  "
          f"Успешность при сохранении: {ckpt.get('success_rate',0)*100:.1f}%\n")

    all_cte_fwd    = []
    all_cte_rev    = []
    all_solved_fwd = []
    all_solved_rev = []
    batch_size     = 10

    print(f"{'Тесты':<12} {'Прямых':>8} {'CTE пр.':>10} {'Обратных':>10} {'CTE обр.':>10} {'CTE общий':>11}")
    print(f"{'-'*65}")

    for i in range(num_tests):
        n_wp = np.random.randint(1, 5)

        #прямой прогон
        env_fwd = TestEnv(n_waypoints=n_wp+2)
        fwd_solved, fwd_path, fwd_cte = _run_episode_dqn(net, env_fwd)

        #обратный прогон
        env_rev = TestEnv.__new__(TestEnv)
        env_rev.stp = env_fwd.stp
        env_rev.av  = env_fwd.av
        env_rev.wps = list(reversed(env_fwd.wps))
        env_rev.wi  = 1
        env_rev.pos = env_rev.wps[0].copy()
        env_rev.path_taken = [env_rev.pos.copy()]
        env_rev.cte_hist   = []
        env_rev.prev_cte   = 0.0
        rev_solved, rev_path, rev_cte = _run_episode_dqn(net, env_rev)

        all_cte_fwd.append(fwd_cte)
        all_cte_rev.append(rev_cte)
        all_solved_fwd.append(fwd_solved)
        all_solved_rev.append(rev_solved)

        if not fwd_solved or not rev_solved:
            _visualize_pair_dqn(env_fwd, fwd_path, fwd_cte, fwd_solved,
                                env_rev, rev_path, rev_cte, rev_solved,
                                i+1, n_wp)

        if (i + 1) % batch_size == 0:
            batch_start   = i + 1 - batch_size
            b_cte_fwd     = all_cte_fwd[batch_start:i+1]
            b_cte_rev     = all_cte_rev[batch_start:i+1]
            b_cte_all     = b_cte_fwd + b_cte_rev
            batch_fwd     = sum(all_solved_fwd[batch_start:i+1])
            batch_rev     = sum(all_solved_rev[batch_start:i+1])
            label = f"{batch_start+1}–{i+1}"
            print(f"{label:<12} {batch_fwd:>4}/{batch_size:<3} "
                  f"{np.mean(b_cte_fwd):>8.2f} пкс "
                  f"{batch_rev:>4}/{batch_size:<4} "
                  f"{np.mean(b_cte_rev):>8.2f} пкс "
                  f"{np.mean(b_cte_all):>8.2f} пкс")

    all_cte_combined = all_cte_fwd + all_cte_rev
    print(f"{'='*65}")
    print(f"{'Итого':<12} {sum(all_solved_fwd):>4}/{num_tests:<3} "
          f"{np.mean(all_cte_fwd):>8.2f} пкс "
          f"{sum(all_solved_rev):>4}/{num_tests:<4} "
          f"{np.mean(all_cte_rev):>8.2f} пкс "
          f"{np.mean(all_cte_combined):>8.2f} пкс")
    print(f"{'='*65}")

# =====================================================================
# ТЕСТИРОВАНИЕ — test2
# =====================================================================

def run_testing_theta(mpath: str, num_tests: int = 100):
    print(f'\n{"="*60}')
    print(f'ТЕСТИРОВАНИЕ test2 (Theta* + препятствия)')
    print(f'{"="*60}')

    if not os.path.exists(mpath):
        print(f'Файл не найден: {mpath}')
        return

    ckpt = torch.load(mpath, map_location=device, weights_only=False)
    net = DQN(layers_sizes).to(device)
    net.load_state_dict(ckpt['policy_net'])
    net.eval()

    print(f"Эпизод: {ckpt.get('episode','?')}  |  "
          f"Успешность при сохранении: {ckpt.get('success_rate',0)*100:.1f}%\n")

    all_cte    = []
    all_solved = []
    batch_size = 10

    print(f"{'Тесты':<12} {'Успешных':>10} {'Среднее CTE':>14}")
    print(f"{'-'*40}")

    for i in range(num_tests):
        env = ThetaStarTestEnv()
        s   = env.reset()
        st  = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        solved = False

        for _ in range(t_max):
            with torch.no_grad():
                a = net(st).max(1).indices.view(1, 1)
            ns, _, done, solved = env.step(a.item())
            st = torch.tensor(ns, dtype=torch.float32,
                               device=device).unsqueeze(0)
            if done:
                break

        all_cte.append(env.mean_cte())
        all_solved.append(solved)

        if not solved:
            env.visualize(title=f'test2 Тест {i+1}: НЕУДАЧА | '
                                f'WP: {len(env.wps)-2} | '
                                f'CTE: {env.mean_cte():.2f} пкс | '
                                f'Шагов: {len(env.path_taken)}')

        env.visualize(title=f'test2 Тест {i+1}: НЕУДАЧА | '
                                f'WP: {len(env.wps)-2} | '
                                f'CTE: {env.mean_cte():.2f} пкс | '
                                f'Шагов: {len(env.path_taken)}')

        if (i + 1) % batch_size == 0:
            batch_start  = i + 1 - batch_size
            batch_cte    = all_cte[batch_start:i+1]
            batch_solved = sum(all_solved[batch_start:i+1])
            label = f"{batch_start+1}–{i+1}"
            print(f"{label:<12} {batch_solved:>4}/{batch_size:<5} "
                  f"{np.mean(batch_cte):>10.2f} пкс")

    print(f"{'='*40}")
    total_solved = sum(all_solved)
    print(f"{'Итого':<12} {total_solved:>4}/{num_tests:<5} "
          f"{np.mean(all_cte):>10.2f} пкс")
    print(f'{"="*60}')

# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================

if __name__ == '__main__':
    if train_test == 'train':
        rw, sc, cte = run_training()
        plot_curves(rw, sc, cte)
        run_testing(model_path, num_tests=10)
    elif train_test == 'test1':
        run_testing(model_path, num_tests=100)
    elif train_test == 'test2':
        run_testing_theta(model_path, num_tests=100)
    else:
        print(f"Неизвестный режим.'train', 'test1' или 'test2'.")
