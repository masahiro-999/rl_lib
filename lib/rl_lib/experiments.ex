defmodule RlLib.Experiments do

  alias __MODULE__

  defstruct [
    :agent,
    :env,
    :outdir,
    :checkpoint_freq,
    :limit_steps,
    :step_offset,
    :evaluator,
    :successful_score,
    :step_hooks,
    :logger,
    :episode_count,
    :max_episode,
    :total_reward,
    :filename
  ]

  def train_agent(agent, env, max_episode, filename, opts \\ []) do
    # {checkpoint_freq, opts} = Keyword.pop(opts, :checkpoint_freq, nil)
    # {max_episode_len, opts} = Keyword.pop(opts, :max_episode_len, nil)
    # {step_offset, opts} = Keyword.pop(opts, :step_offset, nil)
    # {evaluator, opts} = Keyword.pop(opts, :evaluator, nil)
    # {successful_score, opts} = Keyword.pop(opts, :successful_score, nil)
    # {step_hooks, opts} = Keyword.pop(opts, :step_hooks, nil)
    # {logger, opts} = Keyword.pop(opts, :logger, nil)

    experiments =
      struct!(
        Experiments,
        [agent: agent, env: env, max_episode: max_episode, filename: filename, episode_count: 0] ++
          opts
      )

    experiments =start_of_episode(experiments)
    if experiments.filename != nil, do: RlLib.Agent.save(experiments.agent, experiments.filename)
    experiments.agent
  end

  def eval_agent(agent, env, max_episode, opts \\ []) do
    agent = struct!(agent, training: false)
    train_agent(agent, env, max_episode, nil, opts)
  end

  def start_of_episode(
        %Experiments{max_episode: max_episode, episode_count: episode_count} = experiments
      )
      when max_episode <= episode_count do
    experiments
  end

  def start_of_episode(%Experiments{env: env, episode_count: episode_count} = experiments) do
    {env, state, _info} = RlLib.Env.reset(env)

    experiments
    |> struct!(env: env, episode_count: episode_count + 1)
    |> episode_loop(state)
    |> start_of_episode()
  end

  def episode_loop(experiments, state, reward \\ 0, total_reward \\ 0, done \\ false)

  def episode_loop(%Experiments{agent: agent} = experiments, state, reward, total_reward, true) do
    agent = RlLib.Agent.stop_episode_and_train(agent, state, reward, true)

    experiments
    |> struct!(agent: agent, total_reward: total_reward)
    |> call_step_hooks()
  end

  def episode_loop(
        %Experiments{agent: agent, env: env} = experiments,
        state,
        reward,
        total_reward,
        _done
      ) do
    {agent, action} = RlLib.Agent.act_and_train(agent, state, reward)
    {env, reward, next_state, done} = RlLib.Env.step(env, action)

    experiments
    |> struct!(agent: agent, env: env)
    |> episode_loop(next_state, reward, total_reward + reward, done)

  end

  def call_step_hooks(%Experiments{step_hooks: nil} = experiments) do
    experiments
  end

  def call_step_hooks(%Experiments{step_hooks: {hook_fn, hook_param}} = experiments) do
    struct!(experiments, step_hooks: {hook_fn, hook_fn.(hook_param, experiments)})
  end
end
