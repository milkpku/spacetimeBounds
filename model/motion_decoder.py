import torch
import torch.nn as nn

class MotionDecoder(nn.Module):
  """ take feature sequence in, and decode them to motion sequence
      motion sampling frequency should be 60 Hz

      reference:
      A Deep Learning Framework for Character Motion Synthesis and Editing
  """
  def __init__(self):
    super(MotionDecoder, self).__init__()

    self.conv = nn.Conv1d(256, 73, 25, padding=12)
    self.mean = nn.Parameter(torch.zeros((1, 73, 1)))
    self.std = nn.Parameter(torch.ones((1, 73, 1)))

  def forward(self, x):
    y = self.conv(self.depool(x))
    out = y * self.std + self.mean
    return out

  def depool(self, x):
    batch, dim, l = x.shape

    mask = torch.rand(batch, dim, l, 2)
    mask = mask.max(3).indices
    half_out_a = (mask * x).reshape(batch, dim, l, 1)
    half_out_b = ((1-mask) * x).reshape(batch, dim, l, 1)
    out = torch.cat([half_out_a, half_out_b], 3)
    out = out.reshape(batch, dim, 2*l)

    return out

decoder = None

def init_motion_decoder():
  global decoder
  decoder = MotionDecoder()
  import pathlib
  dir_path = pathlib.Path(__file__).parent.absolute()
  decoder.load_state_dict(torch.load("%s/motion_decoder.tar" % dir_path))

def decode_embedding(emb):
  if decoder is None:
    init_motion_decoder()
  with torch.no_grad():
    t = torch.Tensor(emb)
    feature = decoder(t).cpu().numpy()
  return feature

if __name__=="__main__":

  decoder_torch = MotionDecoder()
  decoder_torch.load_state_dict(torch.load("motion_decoder.tar"))

  import numpy as np
  a = np.random.rand(1, 256, 120)
  t = torch.Tensor(a)

  feature = decoder_torch(t)

  from IPython import embed; embed()
