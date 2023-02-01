defprotocol RlLib.Agent do
  def act_and_train(agent, obs, reward)
  def stop_episode_and_train(agent, state, reward, done)
  def save(agent, filename)
end
