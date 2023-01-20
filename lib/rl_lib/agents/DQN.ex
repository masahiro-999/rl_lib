defmodule RlLib.Agents.DQN do

  alias __MODULE__
  alias RlLib.QNet

  import Nx.Defn

  defstruct gamma: 0.98,
            epsilon: 1,
            epsilon_min: 0.01,
            epsilon_decay: 0.9999,
            buffer_size: 10000,
            batch_size: 32,
            action_size: nil,
            lr: nil,
            qnet: nil,
            qnet_target: nil,
            replay_buffer: nil,
            model: nil,
            initial_data: nil,
            is_replay_ready: false,
            info: {-1, -1},
            counter: %{win: 0, miss: 0, lose: 0}

  def new(agent \\ %DQN{}) do
    agent
    |> set_model()
    |> set_replay_buffer()
    |> set_q_net()
    |> sync_q_net()
  end

  def set_model(agent) do
    size = agent.action_size
    model =
      Axon.input("x", shape: {nil, 2,size,size})
      |> Axon.conv(1024, kernel_size: 3, activation: :linear )
      |> Axon.conv(1024, kernel_size: 2, activation: :linear )
      |> Axon.conv(1024, kernel_size: 2, activation: :linear )
      |> Axon.flatten()
      |> Axon.dense(1024, activation: :relu)
      |> Axon.dense(1024, activation: :relu)
      |> Axon.dense(1024, activation: :relu)
      |> Axon.dense(size**2)

    %DQN{agent | model: model}
  end

  def set_replay_buffer(agent) do
    %DQN{agent | replay_buffer: Deque.new(agent.buffer_size)}
  end

  def set_q_net(agent) do
    %DQN{
      agent
      | qnet:
          QNet.new(
            model: agent.model,
            optimizer: Axon.Optimizers.adam(agent.lr)
          )
    }
  end

  def sync_q_net(%DQN{} = agent) do
    %DQN{agent | qnet_target: agent.qnet}
  end

  def epsilon_decay(%DQN{} = agent) do
    %DQN{agent | epsilon: max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)}
  end

  def add_replay_buffer(%DQN{} = agent, data) do
    %DQN{agent | replay_buffer: Deque.appendleft(agent.replay_buffer, data)}
  end

  def get_action(state, %DQN{} = agent, leagal_action) do
    cond do
      :rand.uniform() < agent.epsilon ->
        hd(Enum.take_random(leagal_action, 1))

      true ->
        Nx.tensor([state])
        |> QNet.forward(agent.qnet)
        |> Nx.argmax()
        |> Nx.to_number()
    end
  end

  def update_counter(%DQN{counter: counter} = agent, reward) do
    %{win: win, miss: miss, lose: lose} = counter
    win = if(reward == 1, do: win + 1, else: win)
    miss = if(reward == -1.5, do: miss + 1, else: miss)
    lose = if(reward == -1, do: lose + 1, else: lose)

    struct!(agent,
      counter: %{
        win: win,
        miss: miss,
        lose: lose,
        draw: 100 - win - miss - lose
      }
    )
  end

  def clear_counter(%DQN{} = agent) do
    struct!(agent,
      counter: %{
        win: 0,
        miss: 0,
        lose: 0
      }
    )
  end

  def update(agent, state, action, reward, next_state, done) do
    agent
    |> add_replay_buffer({state, [action], [reward], next_state, [done]})
    |> batch_update()
  end

  def batch_update(%DQN{is_replay_ready: false} = agent) do
    %DQN{agent | is_replay_ready: Enum.count(agent.replay_buffer) >= agent.batch_size}
  end

  def batch_update(%DQN{replay_buffer: replay_buffer, batch_size: batch_size} = agent) do
    batch_data = Enum.take_random(replay_buffer, batch_size)
    {state, action, reword, next_state, done} = to_nx_tensor(batch_data)

    agent
    |> batch_update_sub(state, action, reword, next_state, done)
    |> epsilon_decay()
  end

  def get_one_element(data, index) do
    for d <- data, do: elem(d, index)
  end

  def to_nx_tensor(data) do
    {
      Nx.tensor(get_one_element(data, 0)),
      Nx.tensor(get_one_element(data, 1)),
      Nx.tensor(get_one_element(data, 2)),
      Nx.tensor(get_one_element(data, 3)),
      Nx.tensor(get_one_element(data, 4))
    }
  end

  def batch_update_sub(%DQN{} = agent, state, action, reward, next_state, done) do
    next_q_max =
      next_state
      |> QNet.forward(agent.qnet_target)
      |> Nx.reduce_max(axes: [1], keep_axes: true)

    target = get_target(reward, agent.gamma, next_q_max, done) |> Nx.backend_copy()
    # IO.inspect(target, label: "target")
    # IO.inspect(reward, label: "reward")
    # IO.inspect(state, label: "state")
    # IO.inspect(next_state, label: "next_state")
    # IO.inspect(action, label: "action")
    # IO.inspect(next_q_max, label: "next_q_max")
    %DQN{
      agent
      | qnet: QNet.backword(agent.qnet, {state, action, target}),
        info: QNet.get_info(agent.qnet)
    }
  end

  defn get_target(reward, gamma, next_q_max, done) do
    reward + (1 - done) * gamma * next_q_max
  end
end
