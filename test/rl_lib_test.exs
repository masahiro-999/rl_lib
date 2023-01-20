defmodule RlLibTest do
  use ExUnit.Case
  doctest RlLib

  test "greets the world" do
    assert RlLib.hello() == :world
  end
end
