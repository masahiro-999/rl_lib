defmodule RlLib.Explorers.EpsilonGreedy do
  alias __MODULE__

  defstruct [:start_epsilon, :end_epsilon, :decay_steps, :random_action]

  def new(start_epsilon, end_epsilon, decay_steps, random_action) do
    struct!(EpsilonGreedy,
      start_epsilon: start_epsilon,
      end_epsilon: end_epsilon,
      decay_steps: decay_steps,
      random_action: random_action
    )
  end

  def get_epsilon(
        %EpsilonGreedy{
          start_epsilon: start_epsilon,
          end_epsilon: end_epsilon,
          decay_steps: decay_steps
        },
        t
      ) do
    d = start_epsilon - end_epsilon
    start_epsilon - d / decay_steps * t
  end

  defimpl RlLib.Explorer do
    def select_action(
          %EpsilonGreedy{random_action: random_action} = epsilon_greedy,
          t,
          greedy_action_func,
          param \\ nil
        )
        when is_function(greedy_action_func) do
      cond do
        :rand.uniform() < EpsilonGreedy.get_epsilon(epsilon_greedy, t) ->
          EpsilonGreedy.get_random_action(random_action, param)

        true ->
          EpsilonGreedy.call_function(greedy_action_func, param)
      end
    end
  end

  def call_function(func, param \\ nil)
  def call_function(func, param) when is_function(func, 1), do: func.(param)
  def call_function(func, _param) when is_function(func, 0), do: func.()

  def get_random_action(random_action, param \\ nil)

  def get_random_action(random_action, param) when is_function(random_action),
    do: call_function(random_action, param)

  def get_random_action(random_action, _param)
      when is_integer(random_action) and random_action > 0,
      do: get_random_action(0..(random_action - 1))

  def get_random_action(random_action, _param), do: hd(Enum.take_random(random_action, 1))

  def set_random_action(%EpsilonGreedy{} = epsilon_greedy, random_action) do
    struct!(epsilon_greedy, random_action: random_action)
  end
end
