defmodule RlLib.Explorers.EpsilonGreedyExponentialDecay do
  alias __MODULE__
  alias RlLib.Explorers.EpsilonGreedy

  defstruct [:start_epsilon, :min_epsilon, :decay, :random_action]

  def new(start_epsilon, min_epsilon, decay_step, random_action) do
    decay = :math.exp(:math.log(min_epsilon / start_epsilon) / decay_step)

    struct!(EpsilonGreedyExponentialDecay,
      start_epsilon: start_epsilon,
      min_epsilon: min_epsilon,
      decay: decay,
      random_action: random_action
    )
  end

  def get_epsilon(
        %EpsilonGreedyExponentialDecay{
          start_epsilon: start_epsilon,
          min_epsilon: min_epsilon,
          decay: decay
        },
        t
      ) do
    max(min_epsilon, start_epsilon * decay ** t)
  end

  defimpl RlLib.Explorer do
    def select_action(
          %EpsilonGreedyExponentialDecay{random_action: random_action} = epsilon_greedy,
          t,
          greedy_action_func,
          param \\ nil
        )
        when is_function(greedy_action_func) do
      cond do
        :rand.uniform() < EpsilonGreedyExponentialDecay.get_epsilon(epsilon_greedy, t) ->
          EpsilonGreedy.get_random_action(random_action, param)

        true ->
          EpsilonGreedy.call_function(greedy_action_func, param)
      end
    end
  end
end
