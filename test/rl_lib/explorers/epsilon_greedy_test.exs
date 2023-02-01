defmodule RlLib.Explorers.EpsilonGreedyTest do
  use ExUnit.Case

  alias RlLib.Explorers.EpsilonGreedy
  alias RlLib.Explorer

  test "new" do
    eg = EpsilonGreedy.new(0.9, 0.1, 100, 2)

    assert %EpsilonGreedy{
             start_epsilon: 0.9,
             end_epsilon: 0.1,
             decay_steps: 100,
             random_action: 2
           } = eg
  end

  test "select random" do
    random_action_func = fn -> :random end
    greedy_action_func = fn -> :greedy end
    eg = EpsilonGreedy.new(1, 1, 100, random_action_func)

    for _i <- 1..10 do
      assert :random ==
               RlLib.Explorer.select_action(eg, 10, greedy_action_func)
    end
  end

  test "select greedy" do
    random_action_func = fn -> :random end
    greedy_action_func = fn -> :greedy end
    eg = EpsilonGreedy.new(0, 0, 100, random_action_func)

    for _i <- 1..10 do
      assert :greedy ==
               Explorer.select_action(eg, 100, greedy_action_func)
    end
  end

  test "select decay" do
    random_action_func = fn -> :random end
    greedy_action_func = fn -> :greedy end
    eg = EpsilonGreedy.new(1, 0, 100, random_action_func)

    for _i <- 1..10 do
      assert :random == Explorer.select_action(eg, 0, greedy_action_func)
    end

    for _i <- 1..10 do
      assert :greedy ==
               Explorer.select_action(eg, 100, greedy_action_func)
    end
  end

  test "random_action func/0" do
    func = fn -> :a end
    assert EpsilonGreedy.get_random_action(func) == :a
  end

  test "random_action func/1" do
    func = fn val -> val end
    assert EpsilonGreedy.get_random_action(func, :a) == :a
  end

  test "random_action integer" do
    result = for _i <- 1..10, do: EpsilonGreedy.get_random_action(2)
    assert Enum.all?(result, fn val -> val in [0, 1] end)
    assert Enum.any?(result, fn val -> val == 0 end)
    assert Enum.any?(result, fn val -> val == 1 end)
  end

  test "random_action list" do
    result = for _i <- 1..10, do: EpsilonGreedy.get_random_action(0..1)
    assert Enum.all?(result, fn val -> val in [0, 1] end)
    assert Enum.any?(result, fn val -> val == 0 end)
    assert Enum.any?(result, fn val -> val == 1 end)
  end

  test "random with param" do
    random_action_func = fn x -> x + 1 end
    greedy_action_func = fn x -> x + 2 end
    eg = EpsilonGreedy.new(1, 1, 100, random_action_func)

    for _i <- 1..10 do
      assert 11 ==
               RlLib.Explorer.select_action(eg, 10, greedy_action_func, 10)
    end
  end

  test "greedy with param" do
    random_action_func = fn x -> x + 1 end
    greedy_action_func = fn x -> x + 2 end
    eg = EpsilonGreedy.new(0, 0, 100, random_action_func)

    for _i <- 1..10 do
      assert 12 ==
               RlLib.Explorer.select_action(eg, 10, greedy_action_func, 10)
    end
  end
end
