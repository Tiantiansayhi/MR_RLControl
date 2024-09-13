from stable_baselines3 import PPO
from MREnv import MREnv
import time
# 创建环境实例
env = MREnv()

# 定义 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)

# 开始训练
model.learn(total_timesteps=10000)

# 保存模型
loca = time.time
model.save("model\model-"+str(loca))

# 加载模型并测试
model = PPO.load("model-")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
        print(f"Episode finished after {i+1} timesteps")
