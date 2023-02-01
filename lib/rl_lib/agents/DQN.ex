defmodule RlLib.Agents.DQN do
  alias __MODULE__
  alias RlLib.QNet
  alias RlLib.Explorer
  alias RlLib.QNet

  import Nx.Defn
  import Nx

  defstruct gamma: nil,
            target_update_interval: 10000,
            buffer_size: 10000,
            batch_size: 32,
            episode_count: 0,
            is_replay_ready: false,
            last_state: nil,
            last_action: nil,
            q_net: nil,
            q_net_target: nil,
            replay_buffer: nil,
            explorer: nil,
            training: true

  def new(opts \\ []) do
    {q_function, opts} = Keyword.pop(opts, :q_function, nil)
    {optimizer, opts} = Keyword.pop(opts, :optimizer, nil)
    {buffer_size, opts} = Keyword.pop(opts, :buffer_size, 10000)
    {target_update_interval, opts} = Keyword.pop(opts, :target_update_interval, 10000)
    {batch_size, opts} = Keyword.pop(opts, :batch_size, 32)
    {gamma, opts} = Keyword.pop(opts, :gamma, 0.9)
    {explorer, _opts} = Keyword.pop(opts, :explorer, nil)

    q_net = QNet.new(q_function, optimizer)

    struct!(DQN,
      q_net: q_net,
      q_net_target: q_net,
      target_update_interval: target_update_interval,
      explorer: explorer,
      replay_buffer: Deque.new(buffer_size),
      batch_size: batch_size,
      gamma: gamma
    )
  end

  def update_target_if_necessary(%DQN{q_net: q_net} = agent) do
    cond do
      is_target_sync_required?(agent) ->
        struct!(agent, q_net_target: q_net)

      true ->
        agent
    end
  end

  def is_target_sync_required?(%DQN{} = agent) do
    Integer.mod(agent.episode_count, agent.target_update_interval) == 0
  end

  def add_replay_buffer(%DQN{} = agent, data) do
    %DQN{agent | replay_buffer: Deque.appendleft(agent.replay_buffer, data)}
  end

  def greedy_action({state, %DQN{} = agent}) when is_list(state) do
    greedy_action({Nx.tensor(state), agent})
  end

  def greedy_action({state, %DQN{} = agent}) when is_tensor(state) do
    state
    |> Nx.new_axis(0)
    |> QNet.forward(agent.q_net)
    |> Nx.argmax()
    |> Nx.to_number()
  end

  def get_action(state, %DQN{explorer: explorer, episode_count: episode_count, training: training} = agent) do
    if training do
      Explorer.select_action(explorer, episode_count, &greedy_action/1, {state, agent})
    else
      greedy_action({state, agent})
    end
  end

  def update(%DQN{} = agent, _state, nil = _action, _reward, _next_state, _done) do
    # do nothing if no action.
    agent
  end

  def update(%DQN{} = agent, state, action, reward, next_state, done) do
    agent
    |> DQN.add_replay_buffer({state, [action], [reward], next_state, [done]})
    |> DQN.batch_update()
  end

  def batch_update(%DQN{is_replay_ready: false} = agent) do
    struct!(agent, is_replay_ready: Enum.count(agent.replay_buffer) >= agent.batch_size)
  end

  def batch_update(%DQN{replay_buffer: replay_buffer, batch_size: batch_size} = agent) do
    batch_data = Enum.take_random(replay_buffer, batch_size)
    {state, action, reword, next_state, done} = to_nx_tensor(batch_data)
    batch_update_sub(agent, state, action, reword, next_state, done)
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
      |> QNet.forward(agent.q_net_target)
      |> Nx.reduce_max(axes: [1], keep_axes: true)

    target = get_target(reward, agent.gamma, next_q_max, done) |> Nx.backend_copy()

    struct!(agent, q_net: QNet.backword(agent.q_net, {state, action, target}))
  end

  def end_of_episode(%DQN{episode_count: episode_count} = agent) do
    agent
    |> struct!(episode_count: episode_count + 1, last_state: nil, last_action: nil)
    |> update_target_if_necessary()
  end

  defn get_target(reward, gamma, next_q_max, done) do
    reward + (1 - done) * gamma * next_q_max
  end

  defimpl RlLib.Agent do
    def act_and_train(%DQN{last_state: last_state, last_action: last_action, training: training} = agent, obs, reward) do
      agent = if training, do: DQN.update(agent, last_state, last_action, reward, obs, false), else: agent
      action = DQN.get_action(obs, agent)
      agent = struct!(agent, last_state: obs, last_action: action)

      {agent, action}
    end

    def stop_episode_and_train(
          %DQN{last_state: last_state, last_action: last_action} = agent,
          state,
          reward,
          done
        ) do
      agent
      |> DQN.update(last_state, last_action, reward, state, done)
      |> DQN.end_of_episode()
    end

    def save(%DQN{q_net: q_net}, filename) do
      QNet.save_model_state(q_net, "#{filename}.axon")
    end
  end
end
