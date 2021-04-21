import numpy as np

class MotionBound(object):
  """ Motion bound to be applied on motions
  """

  def __init__(self, bound):
    """
      Inputs:

        bound   np.array of shape [n, dim], where n is the number of frames,
                corresponding to uniform frames in [0, 1] (including 0 and 1),
                dim is the
    """
    self._bound = bound
    self._divs = bound.shape[0] - 1

  def interp(self, phase):
    """ Get corresponding motion bound for phase p in [0, 1]
      Inputs:
        phase   float, phase of motion
    """
    phase = phase % 1
    phase = phase * self._divs
    idx = int(phase)

    if idx == self._divs:
      idx -= 1

    phase = phase - idx
    bound = self._bound[idx] * (1-phase) + self._bound[idx+1] * phase

    return bound

def build_motion_bound(bound_file):
  """ If bound is float, assign all to float,
  """
  assert(type(bound_file) is str)

  frames = np.loadtxt(bound_file)
  motion_bound = MotionBound(frames)

  return motion_bound

def design_motion_bound(phases, frames, divs):
  xvals = np.linspace(0, 1, divs+1)
  yvals = list(map(lambda i: np.interp(xvals, phases, frames[:, i]), range(frames.shape[1])))
  yvals = np.array(yvals)
  bounds = yvals.transpose()

  return bounds

if __name__=="__main__":
  from IPython import embed

  N = 20
  dim = 30
  duration = 15

  k = np.random.rand(dim)
  frames = np.zeros((N, dim))

  for i in range(N):
    frames[i] = k * i * duration / (N-1)

  bound = MotionBound(frames)

  t_array = np.random.rand(30) * duration
  for t in t_array:
    b = bound.interp(t/duration)

    diff = np.linalg.norm(t*k -b)
    assert(abs(diff) < 1e-9), diff

  # test design_motion_bound
  phases = [0, 0.2, 0.8, 1]
  bound = np.ones(20)
  frames = []
  frames.append(bound.copy() * 0.3)
  frames.append(bound.copy() * 0.7)
  frames.append(bound.copy() * 0.7)
  frames.append(bound.copy() * 0.3)
  frames = np.array(frames)

  bounds = design_motion_bound(phases, frames, 10)
  print(bounds)
  embed()
