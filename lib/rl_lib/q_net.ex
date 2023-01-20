defmodule RlLib.QNet do

  alias __MODULE__

  defstruct [:step_state, :model, :init_fn, :step_fn]

  def build_model() do
    Axon.input("x", shape: {nil, 12})
    |> Axon.dense(100)
    |> Axon.relu()
    |> Axon.dense(4)
  end

  def new(opts \\ []) do
    {model, opts} = Keyword.pop(opts, :model, build_model())
    {optimizer, _opts} = Keyword.pop(opts, :optimizer, Axon.Optimizers.sgd())

    {init_fn, step_fn} =
      train_step(
        model,
        &Axon.Losses.mean_squared_error(&1, &2, reduction: :mean),
        optimizer
      )

    %QNet{
      step_state: nil,
      model: model,
      step_fn: step_fn,
      init_fn: init_fn
    }
  end

  def init_step_state(%QNet{step_state: nil} = q_net, inp) do
    %QNet{q_net | step_state: q_net.init_fn.({inp, 0}, %{})}
  end

  def backword(%QNet{step_state: nil} = q_net, {inp, a, tar}) do
    backword(init_step_state(q_net, inp), {inp, a, tar})
  end

  def backword(%QNet{} = q_net, {inp, a, tar}) do
    new_step_state = q_net.step_fn.({%{"x" => inp}, a, tar}, q_net.step_state)
    %QNet{q_net | step_state: new_step_state}
  end

  def forward(inp, %QNet{step_state: nil} = q_net) do
    forward(inp, init_step_state(q_net, inp))
  end

  def forward(inp, %QNet{} = q_net) do
    Axon.predict(q_net.model, q_net.step_state.model_state, %{"x" => inp})
  end

  def train_step(model, loss_fn, {init_optimizer_fn, update_optimizer_fn}) do
    {init_model_fn, forward_model_fn} = Axon.build(model, mode: :train)

    init_fn = fn {inp, _}, init_model_state ->
      model_state = init_model_fn.(inp, init_model_state)
      optimizer_state = init_optimizer_fn.(model_state)

      %{
        i: Nx.tensor(0),
        y_true: Nx.tensor(0.0),
        y_pred: Nx.tensor(0.0),
        loss: Nx.tensor(0.0),
        model_state: model_state,
        optimizer_state: optimizer_state
      }
    end

    objective_fn = fn state, inp, a, tar ->
      model_out = forward_model_fn.(state, inp)
      model_out = %{model_out | prediction: Nx.take_along_axis(model_out.prediction, a, axis: 1)}
      {model_out, loss_fn.(tar, model_out.prediction)}
    end

    step_fn = fn {inp, a, tar}, state ->
      %{i: i, model_state: model_state, optimizer_state: optimizer_state, loss: loss} = state

      {{model_out, batch_loss}, gradients} =
        Nx.Defn.value_and_grad(
          model_state,
          &objective_fn.(&1, inp, a, tar),
          fn x -> elem(x, 1) end
        )

      preds = model_out.prediction
      new_state = model_out.state

      new_loss =
        loss
        |> Nx.multiply(i)
        |> Nx.add(batch_loss)
        |> Nx.divide(Nx.add(i, 1))

      {updates, new_optimizer_state} =
        update_optimizer_fn.(gradients, optimizer_state, model_state)

      new_model_state = Axon.Updates.apply_updates(model_state, updates, new_state)

      %{
        state
        | i: Nx.add(i, 1),
          y_true: tar,
          y_pred: preds,
          loss: new_loss,
          model_state: new_model_state,
          optimizer_state: new_optimizer_state
      }
    end

    {
      Nx.Defn.jit(init_fn, on_conflict: :reuse),
      Nx.Defn.jit(step_fn, on_conflict: :reuse)
    }
  end

  def get_info(%QNet{step_state: nil}) do
    {-1, -1}
  end

  def get_info(%QNet{step_state: step_state}) do
    {Nx.to_number(step_state.i), Nx.to_number(step_state.loss)}
  end

  def save_model_state(%QNet{model: model, step_state: step_state}, filename) do
    serialized = Axon.serialize(model, step_state.model_state)
    File.write(filename, serialized)
  end

  def load_model_state(filename) do
    {model, model_state} =
      filename
      |> File.read!()
      |> Axon.deserialize()

    struct!(QNet, model: model, step_state: %{model_state: model_state})
  end
end
