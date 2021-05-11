import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

######
INIT_ACTOR_SCALE = 0.001
NOISE = 0.1
USE_ELU = False

NORM_SAMPLES    = 1000000
######

class Normalizer(nn.Module):
  def __init__(self, in_dim, non_norm, sample_lim=NORM_SAMPLES):
    super(Normalizer, self).__init__()

    self.mean    = nn.Parameter(torch.zeros([in_dim]))
    self.std     = nn.Parameter(torch.ones([in_dim]))
    self.mean_sq = nn.Parameter(torch.ones([in_dim]))
    self.num     = nn.Parameter(torch.zeros([1]))

    self.non_norm = non_norm

    self.sum_new    = nn.Parameter(torch.zeros([in_dim]))
    self.sum_sq_new = nn.Parameter(torch.zeros([in_dim]))
    self.num_new    = nn.Parameter(torch.zeros([1]))

    for param in self.parameters():
      param.requires_grad = False

    self.sample_lim = sample_lim

  def forward(self, x):
    return (x - self.mean) / self.std

  def unnormalize(self, x):
    return x * self.std + self.mean

  def set_mean_std(self, mean, std):
    self.mean.data = torch.Tensor(mean)
    self.std.data = torch.Tensor(std)

  def record(self, x):
    if (self.num + self.num_new >= self.sample_lim):
      return

    if x.dim() == 1:
      self.num_new += 1
      self.sum_new += x
      self.sum_sq_new += torch.pow(x, 2)
    elif x.dim() == 2:
      self.num_new += x.shape[0]
      self.sum_new += torch.sum(x, dim=0)
      self.sum_sq_new += torch.sum(torch.pow(x, 2), dim=0)
    else:
      assert(False and "normalizer record more than 2 dim")

  def update(self):
    if self.num >= self.sample_lim or self.num_new == 0:
      return

    # update mean, mean_sq and std
    total_num = self.num + self.num_new;
    self.mean.data *= (self.num / total_num)
    self.mean.data += self.sum_new / total_num
    self.mean_sq.data *= (self.num / total_num)
    self.mean_sq.data += self.sum_sq_new / total_num
    self.std.data = torch.sqrt(torch.abs(self.mean_sq.data - torch.pow(self.mean.data, 2)))
    self.std.data += 0.01 # in case of divide by 0
    self.num.data += self.num_new

    # NOTICE hack method
    # only the head parameters representing phase should not be normalized
    for i in self.non_norm:
      self.mean.data[i] = 0
      self.std.data[i] = 1.0

    # clear buffer
    self.sum_new.data.zero_()
    self.sum_sq_new.data.zero_()
    self.num_new.data.zero_()

    return

# initialize fc layer using xavier uniform initialization
def xavier_init(module):
  nn.init.xavier_uniform_(module.weight.data, gain=1)
  nn.init.constant_(module.bias.data, 0)
  return module

class Actor(nn.Module):
  def __init__(self, s_dim, a_dim, a_min, a_max, a_noise, mem_size, hidden=[1024, 512]):
    super(Actor, self).__init__()

    # initialize reference memory
    self.clip_size = mem_size-1
    self.ref_mem = nn.Parameter(torch.zeros(mem_size, a_dim))

    # initialize feedback network
    self.fc = []
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(xavier_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)

    # initialize final layer weight to be [INIT, INIT]
    self.fca = nn.Linear(input_dim, a_dim)
    nn.init.uniform_(self.fca.weight, -INIT_ACTOR_SCALE, INIT_ACTOR_SCALE)
    nn.init.constant_(self.fca.bias, 0)

    # set a_norm not trainable
    self.a_min = nn.Parameter(torch.tensor(a_min).float())
    self.a_max = nn.Parameter(torch.tensor(a_max).float())
    std = (a_max - a_min) / 2
    self.a_std = nn.Parameter(torch.tensor(std).float())
    self.a_noise = nn.Parameter(torch.tensor(a_noise).float())
    self.a_min.requires_grad = False
    self.a_max.requires_grad = False
    self.a_std.requires_grad = False
    self.a_noise.requires_grad = False

    self.activation = F.elu if USE_ELU else F.relu

  def set_reference(self, ref_mem):
    assert(ref_mem.shape == self.ref_mem.shape)
    self.ref_mem.data = torch.tensor(ref_mem).float()

  def interp(self, x):
    # reference memory
    if x.dim() == 2:
      phase = x[:, 0]
    else:
      phase = x[0]

    phase = phase.detach()
    idx = torch.floor(phase * self.clip_size)
    idx = torch.clamp(idx, 0, self.clip_size-1)
    t = phase * self.clip_size - idx
    if t.dim() == 1:
      t = t.unsqueeze(1)
    idx = idx.long()
    off_base = self.ref_mem[idx] * (1-t) + self.ref_mem[idx+1] * t
    return off_base

  def feedback(self, x):
    # feedback
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))

    # unnormalize action
    layer_a = self.fca(layer)
    a_offset = self.a_std * layer_a

    return a_offset

  def forward(self, x):
    # noramlize x first

    off_base = self.interp(x)
    a_offset = self.feedback(x)
    a_mean = off_base + a_offset

    return a_mean

  def act_distribution(self, x):
    a_mean = self.forward(x)
    m = D.Normal(a_mean, self.a_noise.view(-1))
    return m

  def act_deterministic(self, x):
    return self.forward(x)

  def act_stochastic(self, x):
    m = self.act_distribution(x)
    ac = m.sample()
    return ac

class Critic(nn.Module):

  def __init__(self, s_dim, val_min, val_max, hidden=[1024, 512]):
    super(Critic, self).__init__()
    self.fc = []
    input_dim = s_dim
    for h_dim in hidden:
      self.fc.append(xavier_init(nn.Linear(input_dim, h_dim)))
      input_dim = h_dim

    self.fc = nn.ModuleList(self.fc)
    self.fcv = xavier_init(nn.Linear(input_dim, 1))

    # value normalizer
    self.v_min = torch.Tensor([val_min])
    self.v_max = torch.Tensor([val_max])
    self.v_mean = nn.Parameter((self.v_max + self.v_min) / 2)
    self.v_std  = nn.Parameter((self.v_max - self.v_min) / 2)
    self.v_min.requires_grad = False
    self.v_max.requires_grad = False
    self.v_mean.requires_grad = False
    self.v_std.requires_grad = False

    self.activation = F.elu if USE_ELU else F.relu

  def forward(self, x):
    layer = x
    for fc_op in self.fc:
      layer = self.activation(fc_op(layer))

    # unnormalize value
    value = self.fcv(layer)
    value = self.v_std * value + self.v_mean

    return value

def load_model(ckpt):
  data = torch.load(ckpt)

  # get info from ckpt, then build network
  s_dim = data["actor"]["fc.0.weight"].shape[1]
  a_dim = data["actor"]["fca.bias"].shape[0]
  a_min = data["actor"]["a_min"].numpy()
  a_max = data["actor"]["a_max"].numpy()
  a_noise = data["actor"]["a_noise"].numpy()
  mem_size = data["actor"]["ref_mem"].shape[0]
  a_hidden = list(map(lambda i: data["actor"]["fc.%d.bias" % i].shape[0], [0, 1]))
  c_hidden = list(map(lambda i: data["critic"]["fc.%d.bias" % i].shape[0], [0, 1]))

  # build network framework
  s_norm = Normalizer(s_dim, non_norm=[], sample_lim=-1) # TODO auto config non_norm
  actor = Actor(s_dim, a_dim, a_min, a_max, a_noise, mem_size, a_hidden)
  critic= Critic(s_dim, 0, 1, c_hidden)

  # load checkpoint
  actor.load_state_dict(data["actor"])
  critic.load_state_dict(data["critic"])
  s_state = s_norm.state_dict()
  s_state.update(data["s_norm"])
  s_norm.load_state_dict(s_state)

  print("load from %s" % ckpt)

  # output
  model = {
      "actor": actor,
      "critic": critic,
      "s_norm": s_norm,
      }

  return model

def unittest():
  import numpy as np

  STATE_DIM = 197
  NON_NORM = [0, 1]
  ACTION_DIM = 36
  MEM_SIZE = 20
  TEST_NUM = 10000

  ACTION_MIN = np.array([-1]*ACTION_DIM)
  ACTION_MAX = np.array([1]*ACTION_DIM)
  ACTION_STD = np.random.rand(ACTION_DIM) + 2

  # Test Normalizer
  s_norm = Normalizer(STATE_DIM, NON_NORM, 10000)

  data = np.random.rand(TEST_NUM, STATE_DIM)
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0) + 0.01
  mean[0] = 0
  mean[1] = 0
  std[0] = 1
  std[1] = 1

  tensor = torch.Tensor(data)

  ## test forward
  assert((tensor == s_norm(tensor)).all())

  # test update
  seq = np.random.rand(20)
  seq *= TEST_NUM / (sum(seq) + 0.1)
  seq = np.around(seq).astype(int)
  last_seq = TEST_NUM - sum(seq)
  seq = seq.tolist()
  seq.append(last_seq)
  offset = 0
  for epoch in seq:
    for i in range(epoch):
      s_norm.record(tensor[offset])
      offset += 1
    s_norm.update()

  mean_torch = s_norm.mean.data.numpy()
  std_torch = s_norm.std.data.numpy()
  assert(np.max(np.abs(mean_torch - mean)) < 1e-6 )
  assert(np.max(np.abs(std_torch  - std )) < 1e-6 )

  ## test another forward
  data_normed = (data - mean)/std
  data_normed_torch = s_norm(tensor).data.numpy()
  assert(np.max(np.abs(data_normed_torch - data_normed )) < 1e-5 )

  ## test normalizer exceed sample count
  s_norm = Normalizer(STATE_DIM, NON_NORM, 100)
  offset = 0
  for epoch in seq:
    for i in range(epoch):
      s_norm.record(tensor[offset])
      offset += 1
    s_norm.update()

  assert(s_norm.num <= s_norm.sample_lim)

  ## test batch record
  s_norm_dummy = Normalizer(STATE_DIM, NON_NORM, 10000)
  data = np.random.rand(TEST_NUM, STATE_DIM)
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0) + 0.01
  for i in NON_NORM:
   mean[i] = 0
   std[i] = 1

  tensor = torch.Tensor(data)
  s_norm_dummy.record(tensor)
  s_norm_dummy.update()
  mean_torch = s_norm_dummy.mean.data.numpy()
  std_torch = s_norm_dummy.std.data.numpy()
  assert(np.max(np.abs(mean_torch - mean)) < 1e-5 )
  assert(np.max(np.abs(std_torch  - std )) < 1e-5 )

  # Test Actor
  ref_mem = np.random.rand(MEM_SIZE, ACTION_DIM)
  ref_mem = ref_mem.astype(np.float32)
  actor = Actor(STATE_DIM, ACTION_DIM, ACTION_MIN, ACTION_MAX, ACTION_STD, MEM_SIZE)
  actor.set_reference(ref_mem)
  m = actor.act_distribution(tensor)

  phase = tensor[:, 0]
  phase = phase.detach()
  clip_size = MEM_SIZE - 1
  idx = np.floor(phase * clip_size)
  idx = np.clip(idx, 0, clip_size-1)
  t = phase * clip_size - idx
  t = t.unsqueeze(1)
  idx = idx.long()
  ref_mem = torch.tensor(ref_mem)
  off_base = ref_mem[idx] * (1-t) + ref_mem[idx+1] * t

  normed_ac = (m.mean - off_base) / actor.a_std
  normed_ac = normed_ac.data.numpy()
  one_sigma = np.sum(np.abs(normed_ac) > INIT_ACTOR_SCALE) /  (TEST_NUM * ACTION_DIM)

if __name__=="__main__":
  unittest()
