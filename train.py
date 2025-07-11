import gymnasium as gym
import torch
from deep_q_learning import Dqn  # Importa seu agente reutilizável!

## ================== 1. CONFIGURAÇÃO DO AMBIENTE E DO AGENTE ==================

# Carrega o ambiente 'MountainCar-v0' do Gymnasium
# 'render_mode="human"' abre uma janela para você assistir. Para treinar rápido, remova este parâmetro.
env = gym.make("MountainCar-v0")

# Pega as especificações do ambiente para configurar nosso agente
# Mountain Car tem 2 observações (posição, velocidade) -> input_size = 2
input_size = env.observation_space.shape[0]
# Mountain Car tem 3 ações (empurrar à esquerda, nada, empurrar à direita) -> nb_action = 3
nb_action = env.action_space.n

# Cria uma instância do seu cérebro Dqn, agora configurado para o Mountain Car
brain = Dqn(input_size=input_size, nb_action=nb_action, gamma=0.99)

## ================== 2. LOOP DE TREINAMENTO ==================

num_episodes = 500  # Quantos jogos completos vamos rodar

print(f"Iniciando treinamento para o Mountain Car...")
print(f"   - Estado (Input): {input_size} neurônios")
print(f"   - Ações (Output): {nb_action} neurônios")

count_tr = 0
count_trunc = 0

for episode in range(num_episodes):
    # Reseta o ambiente para começar um novo jogo
    state, info = env.reset()

    episode_reward = 0
    last_reward = 0  # Sua função 'update' usa a recompensa do passo anterior
    done = False

    while not done:
        # 1. O agente recebe o estado atual e a recompensa anterior,
        #    aprende com isso e escolhe a próxima ação.
        action_tensor = brain.update(state, last_reward)

        # O método .item() converte o tensor de uma única ação (ex: tensor([2])) para um inteiro (ex: 2)
        action = action_tensor.item()

        # 2. Executa a ação no ambiente
        next_state, reward, terminated, truncated, info = env.step(action)
        #print(next_state)
        if terminated:
            reward = 30
        elif next_state[0] > 0.4:
            reward = 4
        elif next_state[0] > 0.3:
            reward = 3
        elif next_state[0] > 0.0:
            reward = 2
        elif next_state[0] > -0.2:
            reward = 1

        # 4. Verifica se o episódio terminou
        if terminated or truncated:
            print(f'Terminated: {terminated} || Truncated: {truncated}')
        if terminated:
            count_tr += 1
        if truncated:
            count_trunc += 1

        done = terminated or truncated

        # 5. Atualiza as variáveis para a próxima iteração
        state = next_state
        last_reward = reward
        episode_reward += reward

        # Renderiza o ambiente (se render_mode='human')
        env.render()

    print(f"Episódio {episode + 1} concluído com recompensa total: {episode_reward:.2f}")

## ================== 3. FINALIZAÇÃO ==================
print("Treinamento concluído!")
print(f'Acertos: {count_tr} / {num_episodes} e Erros: {count_trunc} / {num_episodes}')

# Salva o cérebro treinado
brain.save()

# Fecha o ambiente e a janela de visualização
env.close()