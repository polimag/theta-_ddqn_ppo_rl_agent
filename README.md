# theta-_ddqn_ppo_rl_agent
Hybrid navigation system for a mobile robot: Theta* path planner + RL agents (DDQN and PPO) for waypoint following. Trained and tested in a grid-based simulation.

## Файлы

### Код
- `9action_DDQN_Theta.py` — агент DDQN с дискретным пространством действий (9 направлений).
  Содержит среды обучения и тестирования, планировщик Theta*, архитектуру сети DQN,
  буфер воспроизведения опыта, цикл обучения и два режима тестирования (test1, test2).

- `360_ppo_Theta.py` — агент PPO с непрерывным пространством действий (угол поворота).
  Содержит те же среды и планировщик, архитектуру ActorCritic, роллаут-буфер,
  GAE-оценку преимущества, цикл обучения и два режима тестирования (test1, test2).

### Модели
- `best_model_dqn.pth` — сохранённые веса обученного агента DDQN.
- `best_model_ppo.pth` — сохранённые веса обученного агента PPO.
