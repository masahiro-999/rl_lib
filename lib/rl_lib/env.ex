defprotocol RlLib.Env do
  def reset(env)
  def step(env, action)
end
