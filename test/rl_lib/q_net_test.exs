defmodule RlLib.QNetTest do
  use ExUnit.Case
  alias RlLib.QNet

  @moduletag timeout: :infinity

  test "new2" do
    model =
      Axon.input("x", shape: {nil, 4})
      |> Axon.dense(128)
      |> Axon.relu()
      |> Axon.dense(128)
      |> Axon.relu()
      |> Axon.dense(2)

    QNet.new(
      model,
      Axon.Optimizers.adam(0.001)
    )
  end

  # test "forward" do
  #   :rand.seed(:exsss, {100, 101, 102})

  #   x =
  #     Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12]])

  #   y = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
  #   q_net = QNet.new()
  #   QNet.forward(x, q_net)
  #   # assert QNet.forward(x) == Nx.tensor([
  #   #   [-2.624209403991699, -2.0369317531585693, 2.141974449157715, 9.313855171203613],
  #   #   [-2.624209403991699, -2.0369317531585693, 2.141974449157715, 9.313855171203613]
  #   #   ])
  # end

  # test "backword" do
  #   x =
  #     Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12]])

  #   y = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
  #   a = Nx.tensor([[1], [3]])
  #   data = Stream.repeatedly(fn -> {x, y} end)
  #   q_net = QNet.new()
  #   QNet.backword(q_net, {x, a, y})
  #   QNet.forward(x, q_net)
  #   # assert QNet.forward(x) == Nx.tensor(
  #   # [
  #   #   [-3.3012332916259766, -95.31354522705078, -6.628816604614258, 3.0473413467407227]
  #   # ])
  # end

  def train(_, 0, q_net) do
    q_net
  end

  def train(data, n, q_net) do
    {{x, a}, y} = hd(Enum.take(data, 1))
    q_net = QNet.backword(q_net, {x, a, y})
    QNet.get_info(q_net) |> IO.inspect(label: "info")
    train(data, n - 1, q_net)
  end

  def testfunc(x, a, param) do
    {w, b} = param[a]
    x * w + b
  end

  def build_model_for_test() do
    Axon.input("x", shape: {nil, 12})
    |> Axon.dense(4)
  end

  test "backword_function_test0" do
    :rand.seed(:exsss, {100, 101, 102})

    param = %{0 => {2, 1}, 1 => {3, -1}, 2 => {-1, 5}, 3 => {-2, 0}}

    data4 =
      Stream.repeatedly(fn ->
        0..3
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = i
          y = testfunc(x, a, param)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    x = 1
    y_list = 0..3 |> Enum.map(&testfunc(x, &1, param))

    q_net =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    ret = train(data4, 200, q_net)

    assert ret.step_state[:model_state]["dense_0"]["bias"][0] |> Nx.to_number() |> Float.round() ==
             1

    assert ret.step_state[:model_state]["dense_0"]["bias"][1] |> Nx.to_number() |> Float.round() ==
             -1

    assert ret.step_state[:model_state]["dense_0"]["bias"][2] |> Nx.to_number() |> Float.round() ==
             5

    assert ret.step_state[:model_state]["dense_0"]["bias"][3] |> Nx.to_number() |> Float.round() ==
             0

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][0]
           |> Nx.to_number()
           |> Float.round() == 2

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][1]
           |> Nx.to_number()
           |> Float.round() == 3

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][2]
           |> Nx.to_number()
           |> Float.round() == -1

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][3]
           |> Nx.to_number()
           |> Float.round() == -2
  end

  test "backword_function_test" do
    :rand.seed(:exsss, {100, 101, 102})

    param = %{0 => {2, 1}, 1 => {3, -1}, 2 => {-1, 5}, 3 => {-2, 0}}

    data10 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    x = 1
    y_list = 0..3 |> Enum.map(&testfunc(x, &1, param))

    q_net =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    ret = train(data10, 200, q_net)

    assert ret.step_state[:model_state]["dense_0"]["bias"][0] |> Nx.to_number() |> Float.round() ==
             1

    assert ret.step_state[:model_state]["dense_0"]["bias"][1] |> Nx.to_number() |> Float.round() ==
             -1

    assert ret.step_state[:model_state]["dense_0"]["bias"][2] |> Nx.to_number() |> Float.round() ==
             5

    assert ret.step_state[:model_state]["dense_0"]["bias"][3] |> Nx.to_number() |> Float.round() ==
             0

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][0]
           |> Nx.to_number()
           |> Float.round() == 2

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][1]
           |> Nx.to_number()
           |> Float.round() == 3

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][2]
           |> Nx.to_number()
           |> Float.round() == -1

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][3]
           |> Nx.to_number()
           |> Float.round() == -2
  end

  test "backword_function_test2" do
    :rand.seed(:exsss, {100, 101, 102})

    param1 = %{0 => {2, 1}, 1 => {3, -1}, 2 => {-1, 5}, 3 => {-2, 0}}
    param2 = %{0 => {3, 2}, 1 => {4, 0}, 2 => {0, 6}, 3 => {-1, 1}}

    data1 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param1)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    data2 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param2)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    x = 1
    y_list = 0..3 |> Enum.map(&testfunc(x, &1, param1))

    q_net1 =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    ret1 = train(data1, 200, q_net1)

    q_net2 =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    ret2 = train(data2, 200, q_net2)

    assert ret1.step_state[:model_state]["dense_0"]["bias"][0] |> Nx.to_number() |> Float.round() ==
             1

    assert ret1.step_state[:model_state]["dense_0"]["bias"][1] |> Nx.to_number() |> Float.round() ==
             -1

    assert ret1.step_state[:model_state]["dense_0"]["bias"][2] |> Nx.to_number() |> Float.round() ==
             5

    assert ret1.step_state[:model_state]["dense_0"]["bias"][3] |> Nx.to_number() |> Float.round() ==
             0

    assert ret1.step_state[:model_state]["dense_0"]["kernel"][0][0]
           |> Nx.to_number()
           |> Float.round() == 2

    assert ret1.step_state[:model_state]["dense_0"]["kernel"][0][1]
           |> Nx.to_number()
           |> Float.round() == 3

    assert ret1.step_state[:model_state]["dense_0"]["kernel"][0][2]
           |> Nx.to_number()
           |> Float.round() == -1

    assert ret1.step_state[:model_state]["dense_0"]["kernel"][0][3]
           |> Nx.to_number()
           |> Float.round() == -2

    assert ret2.step_state[:model_state]["dense_0"]["bias"][0] |> Nx.to_number() |> Float.round() ==
             2

    assert ret2.step_state[:model_state]["dense_0"]["bias"][1] |> Nx.to_number() |> Float.round() ==
             0

    assert ret2.step_state[:model_state]["dense_0"]["bias"][2] |> Nx.to_number() |> Float.round() ==
             6

    assert ret2.step_state[:model_state]["dense_0"]["bias"][3] |> Nx.to_number() |> Float.round() ==
             1

    assert ret2.step_state[:model_state]["dense_0"]["kernel"][0][0]
           |> Nx.to_number()
           |> Float.round() == 3

    assert ret2.step_state[:model_state]["dense_0"]["kernel"][0][1]
           |> Nx.to_number()
           |> Float.round() == 4

    assert ret2.step_state[:model_state]["dense_0"]["kernel"][0][2]
           |> Nx.to_number()
           |> Float.round() == 0

    assert ret2.step_state[:model_state]["dense_0"]["kernel"][0][3]
           |> Nx.to_number()
           |> Float.round() == -1
  end

  test "backword_function_test(copy)" do
    :rand.seed(:exsss, {100, 101, 102})

    param1 = %{0 => {2, 1}, 1 => {3, -1}, 2 => {-1, 5}, 3 => {-2, 0}}
    param2 = %{0 => {3, 2}, 1 => {4, 0}, 2 => {0, 6}, 3 => {-1, 1}}

    data1 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param1)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    data2 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param2)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    x = 1
    y_list = 0..3 |> Enum.map(&testfunc(x, &1, param1))

    q_net1 =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    q_net2 =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    q_net1 = train(data1, 200, q_net1)
    q_net2 = q_net1

    ret1 =
      QNet.forward(Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), q_net1)
      |> Nx.to_flat_list()

    ret2 =
      QNet.forward(Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), q_net2)
      |> Nx.to_flat_list()

    assert ret1 == ret2

    q_net1 = train(data2, 200, q_net1)

    ret11 =
      QNet.forward(Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), q_net1)
      |> Nx.to_flat_list()

    ret3 =
      QNet.forward(Nx.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), q_net2)
      |> Nx.to_flat_list()

    assert ret1 != ret11
    assert ret1 == ret3
    assert ret2 == ret3
  end

  test "save load test" do
    :rand.seed(:exsss, {100, 101, 102})

    param = %{0 => {2, 1}, 1 => {3, -1}, 2 => {-1, 5}, 3 => {-2, 0}}

    data10 =
      Stream.repeatedly(fn ->
        0..10
        |> Enum.reduce(nil, fn i, acc ->
          x = :rand.uniform()
          a = Enum.take_random(0..3, 1) |> hd
          y = testfunc(x, a, param)

          case acc do
            nil ->
              {{Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), Nx.tensor([[a]])},
               Nx.tensor([[y]])}

            _ ->
              {{acc_x, acc_a}, acc_y} = acc

              {{Nx.concatenate([acc_x, Nx.tensor([[x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]),
                Nx.concatenate([acc_a, Nx.tensor([[a]])])},
               Nx.concatenate([acc_y, Nx.tensor([[y]])])}
          end
        end)
      end)

    x = 1
    y_list = 0..3 |> Enum.map(&testfunc(x, &1, param))

    q_net =
      QNet.new(
        build_model_for_test(),
        Axon.Optimizers.sgd(1)
      )

    ret = train(data10, 200, q_net)

    QNet.save_model_state(ret, "savedata.axon")
    ret = QNet.load_model_state("savedata.axon")

    assert ret.step_state[:model_state]["dense_0"]["bias"][0] |> Nx.to_number() |> Float.round() ==
             1

    assert ret.step_state[:model_state]["dense_0"]["bias"][1] |> Nx.to_number() |> Float.round() ==
             -1

    assert ret.step_state[:model_state]["dense_0"]["bias"][2] |> Nx.to_number() |> Float.round() ==
             5

    assert ret.step_state[:model_state]["dense_0"]["bias"][3] |> Nx.to_number() |> Float.round() ==
             0

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][0]
           |> Nx.to_number()
           |> Float.round() == 2

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][1]
           |> Nx.to_number()
           |> Float.round() == 3

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][2]
           |> Nx.to_number()
           |> Float.round() == -1

    assert ret.step_state[:model_state]["dense_0"]["kernel"][0][3]
           |> Nx.to_number()
           |> Float.round() == -2
  end
end
