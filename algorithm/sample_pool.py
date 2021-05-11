import numpy as np

class SamplePool:
  """
    Importance sampling tool

    for each bin, there will be a running average value to address the energy

    when sampling, there is probability to sample uniformly by self._base, and by
    (1-self._base) sampling according to Bolzman distribution, which is proportional
    to exp(-energy/T)
  """
  def __init__(self, segments=10, gamma=0.9, base=0.2, tmp=1):
    """
      Inputs:
        segments    int, number of segments to split [0, 1]
        gamma       float, running average discount factor
        base        float in [0, 1], probability to sample uniformly
        tmp         float, temperature of Bolzman distribution
    """
    self._div = segments
    self._gamma = gamma
    self._base = base
    self._tmp = tmp
    self._val_pool = np.zeros(segments)
    self._prob = None

  def record(self, phase, val):
    """ Record one pair of information (phase, val), but no update sampling prob
      Inputs:
        phase   float in [0, 1], the phase of current value
        val     float
    """

    idxs = np.floor(np.clip(phase * self._div, 0, self._div - 0.01))
    for i in range(self._div):
      mask = (idxs == i)
      # choose from all states to get better select set
      val_i = val[mask]
      if val_i.size > 0:
        m = val_i.mean()
        self._val_pool[i] = self._gamma * self._val_pool[i] + (1-self._gamma) * m

  def sample(self):
    """ Sample a float in [0, 1] according to rule

      Outputs:
        phase   float in [0, 1]
    """
    phase = np.random.choice(self._div, p=self._prob) + np.random.rand()
    phase /= self._div
    return phase

  def update_prob(self):
    """ Update sampling probablity according to current recording
    """
    # negative soft max, the lower value the higher probability
    gap = self._val_pool.max() - self._val_pool.min()
    prob = self._val_pool - self._val_pool.max()
    prob /= self._tmp * (gap+0.01)
    prob = np.exp(-prob)
    prob /= prob.sum()
    prob *= (1-self._base)
    prob = prob + self._base / self._div
    self._prob = prob

if __name__=="__main__":
  ##### test runable #####
  pool = SamplePool()
  phase = np.random.rand(200)
  val = np.random.rand(200)
  pool.record(phase, val)
  pool.update_prob()
  print(pool._prob)

  #### test corner case ####
  pool.record(np.array([]), np.array([]))
  pool.update_prob()
  print(pool._prob)

