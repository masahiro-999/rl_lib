defmodule RlLib.Explorers.EpsilonGreedyExponentialDecayTest do
  use ExUnit.Case

  alias RlLib.Explorers.EpsilonGreedyExponentialDecay
  alias RlLib.Explorers.EpsilonGreedy
  alias RlLib.Explorer

  test "new" do
    eg = EpsilonGreedyExponentialDecay.new(0.9, 0.1, 100, 2)

    assert %EpsilonGreedyExponentialDecay{
             start_epsilon: 0.9,
             min_epsilon: 0.1,
             # 0.9*0.9782673857291712**100 = 0.09999999999999988
             decay: 0.9782673857291712,
             random_action: 2
           } = eg
  end

  test "get_epsilon" do
    eg = EpsilonGreedyExponentialDecay.new(0.9, 0.1, 100, 2)

    assert 0.9 == EpsilonGreedyExponentialDecay.get_epsilon(eg, 0)
    assert 0.1 == EpsilonGreedyExponentialDecay.get_epsilon(eg, 100)
  end
end
