import gymnasium as gym
import imageio
import torch
from deep_q_learning import Dqn

# Cria o ambiente com renderização em RGB para gerar imagens
env = gym.make("MountainCar-v0", render_mode="rgb_array")

input_size = env.observation_space.shape[0]
nb_action = env.action_space.n

# Cria o agente e carrega o modelo treinado
brain = Dqn(input_size=input_size, nb_action=nb_action, gamma=0.99)
brain.load()

# Teste
state, _ = env.reset()
state = torch.Tensor(state).float().unsqueeze(0)
done = False
frames = []
episode_reward = 0

while not done:
    # Salva o frame
    frame = env.render()
    frames.append(frame)

    # Escolhe a ação (sem aprender)
    action_tensor = brain.select_action(state)
    action = action_tensor.item()  # converte o tensor para int

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    state = torch.Tensor(next_state).float().unsqueeze(0)
    episode_reward += reward

env.close()

# Cria GIF se o agente completou a tarefa
if terminated:
    imageio.mimsave("acerto_mountaincar.gif", frames, duration=1/30)  # 30 FPS
    print("GIF salvo como 'acerto_mountaincar.gif'")
else:
    print("O agente não conseguiu alcançar o objetivo ainda. Inicie o train.py para treinar um novo cérebro."
          "\n> Possibilidade: Aumente a quantidade de episódios")
