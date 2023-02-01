defmodule RlLib.Agents.DQNTest do
  use ExUnit.Case

  alias RlLib.Agents.DQN
  alias RlLib.QNet

  setup do
    :rand.seed(:exsss, {100, 101, 102})

    q_function =
      Axon.input("x", shape: {nil, 4})
      |> Axon.dense(2)

    explorer = RlLib.Explorers.EpsilonGreedy.new(0.1, 0, 100, 2)

    agent =
      DQN.new(
        q_function: q_function,
        optimizer: Axon.Optimizers.adam(),
        explorer: explorer,
        target_update_interval: 20,
        buffer_size: 10000,
        gamma: 0.9,
        batch_size: 32
      )

    {:ok, agent: agent}
  end

  test "get_action", %{agent: agent} = state do
    assert DQN.get_action(Nx.tensor([[1, 2, 3, 4]]), agent) in [0, 1]
  end

  test "get_action2", %{agent: agent} = state do
    assert DQN.get_action([1, 2, 3, 4], agent) in 0..8
  end

  test "update", %{agent: agent} = state do
    state = [1, 2, 3, 4]
    next_state = [5, 6, 7, 8]
    reword = 1
    action = 1

    DQN.update(agent, state, action, reword, next_state, false)

    for x <- 1..100, reduce: agent do
      agent ->
        DQN.update(agent, state, action, reword, next_state, false)
    end
  end

  test "get_target" do
    reword = Nx.tensor([[1], [1]])
    gamma = 0.9
    next_q_max = Nx.tensor([[0.1], [0.2]])
    done = Nx.tensor([[0], [1]])

    assert DQN.get_target(reword, gamma, next_q_max, done) |> Nx.to_flat_list() ==
             [1.090000033378601, 1.0]
  end

  test "replay_buffer", %{agent: agent} = state do
    state = [1, 2, 3, 4]
    next_state = [5, 6, 7, 8]
    reword = 1
    action = 1

    assert agent.is_replay_ready == false

    agent =
      for x <- 1..32, reduce: agent do
        agent ->
          DQN.update(agent, state, action, reword, next_state, false)
      end

    assert agent.is_replay_ready == true
  end

  test "save", %{agent: agent} = state do
    state = [1, 2, 3, 4]
    next_state = [5, 6, 7, 8]
    reword = 1
    action = 1

    DQN.update(agent, state, action, reword, next_state, false)

    for x <- 1..100, reduce: agent do
      agent ->
        DQN.update(agent, state, action, reword, next_state, false)
    end
    RlLib.Agent.save(agent, "test")
  end

end
