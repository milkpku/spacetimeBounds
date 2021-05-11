import numpy as np

def select_by_boltzmann(val, num, beta=3):
  """ Select by bolzmann distribution

    Inputs:
      val       values of choices, the higher the better
      num       number of choices to be sampled, if larger than val.size, use val.size
      beta      augment ratio of value
  """
  if num >= val.size:
    return np.arange(val.size)

  val_min = val.min()
  val_max = val.max()
  gap = (val_max - val_min) + 0.01   # in case gap == 0
  prob = np.exp(beta * (val - val_max) / gap)
  prob /= prob.sum()
  select = np.random.choice(val.size, num, replace=False, p=prob)
  return select

class SelectPool:
  def __init__(self, segments=10, capacity=200, decay=0.95, noise=0.05):
    """
      Inputs:
        segments    int, number of segmentss to split [0, 1] phases
        capacity    int, number of states each segment contains
        decay       float in [0, 1], the decay parameter of selected set value
    """
    self._div           = segments
    self._capacity      = capacity
    self._decay         = decay
    self._noise         = noise
    self._state_val     = [None] * segments
    self._val_mean      = [None] * segments
    self._val_var       = [None] * segments
    self._noise_level   = [None] * segments
    self._state         = [None] * segments
    self._state_val_new = [None] * segments
    self._state_new     = [None] * segments

  def record(self, val, phase, states):
    """ Record batch of information (val, states), but do not update selections set

        update _data_new by selecting same capacity of states

      Inputs:
        val     np.array of floats
        phase   np.array of floats in [0, 1], the phase of current value and data
        state   np.array of floats, the data environment need to fully reset its state
    """
    idxs = np.floor(np.clip(phase * self._div, 0, self._div - 0.01))
    for i in range(self._div):
      mask = (idxs == i)
      # choose from all states to get better select set
      # TODO better selection scheme
      val_i = val[mask]
      state_i = states[mask]
      good_n = select_by_boltzmann(val_i, self._capacity)
      self._state_val_new[i] = val_i[good_n]
      self._state_new[i] = state_i[good_n]

  def update(self):
    # update good data selection
    for i in range(self._div):
      if self._state[i] is None:
        self._state[i] = self._state_new[i]
        self._state_val[i] = self._state_val_new[i]
        self._val_mean[i] = self._state_val[i].mean()
        self._val_var[i] = self._state_val[i].std()
        self._noise_level[i] = self._noise
      else:
        self._state_val[i] *= self._decay
        val = np.concatenate([self._state_val[i], self._state_val_new[i]], axis=0)
        states = np.concatenate([self._state[i], self._state_new[i]], axis=0)
        order = np.argsort(val)
        good_n = select_by_boltzmann(val, self._capacity)
        self._state[i] = states[good_n]
        self._state_val[i] = val[good_n]
        self._val_mean[i] = self._state_val[i].mean()
        self._val_var[i] = self._state_val[i].std()
        self._noise_level[i] = self._noise

  def get_select_set(self):
    return self._state

  def get_noise_level(self):
    return self._noise_level

def unittest_select_boltzman():

  # test not enough val
  val = np.random.rand(80)
  select = select_by_boltzmann(val, 200)
  assert(select.sum() == np.arange(val.size).sum())

  # test normal distribution
  val = np.random.rand(500)
  select = select_by_boltzmann(val, 200)
  val_good = val[select]
  assert(val_good.mean() > 1.0/1.8), val_good.mean()

  # test extream distribution
  val = np.array([100, 0, 0, 0])
  select = select_by_boltzmann(val, 10)
  assert(0 in select)

  # test corner cases
  val = np.zeros(10)
  select = select_by_boltzmann(val, 9)
  print(select)

  print("Select by Boltzmann pass")

def unittest_selectpool():
  pool = SelectPool()

  # initialize
  val = np.random.rand(100)
  phase = np.random.rand(100)
  data = np.random.rand(100, 30)

  pool.record(val, phase, data)

  # record
  val = np.random.rand(1000)
  phase = np.random.rand(1000)
  data = np.random.rand(1000, 30)

  pool.record(val, phase, data)

  # samples
  pool.get_select_set()

  print("SelectPool pass")

if __name__=="__main__":
  unittest_select_boltzman()
  unittest_selectpool()
