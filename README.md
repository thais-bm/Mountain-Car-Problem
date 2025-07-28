# Mountain-Car-Problem

Este projeto implementa uma solução de Deep Q-Learning para o clássico problema do Mountain Car, utilizando PyTorch e Gymnasium.

## Descrição

O objetivo do agente é treinar um carro para subir uma montanha, aprendendo a melhor política de ações por meio de aprendizado por reforço profundo (Deep Q-Learning). O agente é treinado e testado no ambiente `MountainCar-v0` do Gymnasium.

## Estrutura do Projeto

- `deep_q_learning.py`: Implementação da rede neural, memória de replay e agente DQN.
- `train.py`: Script para treinar o agente no ambiente Mountain Car.
- `test.py`: Script para testar o agente treinado e gerar um GIF do desempenho.
- `last_brain.pth`: Arquivo salvo do modelo treinado.
- `.gitignore`: Arquivos e pastas ignorados pelo git.
- `LICENSE`: Licença Apache 2.0.

## Como usar

### 1. Instale as dependências

```sh
pip install torch gymnasium imageio
```

É recomendado instalar essas bibliotecas num ambiente virtual!!

#### 1.1. Criando um ambiente virtual  
Passo 1: Abra o terminal ou prompt de comando do seu sistema operacional.
Passo 2: Acesse usando o terminal o caminho do seu projeto da seguinte maneira 
```sh
cd /caminho/do/seu/projeto
```
Passo 3: Use o comando
```sh
python -m venv meu_ambiente_virtual
```
Substitua nome_do_ambiente pelo nome que você escolher para o ambiente virtual.
Passo 4: Ative o seu ambiente virtual

* Windows:
```sh
meu_ambiente_virtual\Scripts\activate
```

* MacOS ou Linux:
```sh
source meu_ambiente_virtual/bin/activate
```

Com o ambiente virtual criado e ativado, execute o comando mostrando no passo 1

### 2. Treine o Agente
Execute o script de treinamento:
```sh
python train.py
```

O agente será treinado por 500 episódios (ajuste em train.py se desejar). O modelo treinado será salvo em last_brain.pth.

### 3. Teste o Agente
Após o treinamento, execute:

```sh
python train.py
```

Se o agente alcançar o objetivo, será gerado um GIF chamado acerto_mountaincar.gif mostrando o desempenho.

## Observações
- Para treinar mais rápido, o ambiente é criado sem renderização. Para visualizar o treinamento, altere render_mode=None para render_mode="human" em train.py.
- Se o agente não conseguir completar a tarefa, aumente o número de episódios em train.py.

## Licença
Este projeto está licenciado sob a Licença Apache 2.0.