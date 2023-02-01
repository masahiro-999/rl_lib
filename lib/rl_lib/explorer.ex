defprotocol RlLib.Explorer do
  def select_action(epsilon_greedy, t, greedy_action_func, param \\ nil)
end
