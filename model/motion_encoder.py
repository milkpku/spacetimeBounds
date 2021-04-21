import torch
import torch.nn as nn

class MotionEncoder(nn.Module):
  """ take motion sequence in, and encode them to latent space
      motion sampling frequency should be 60 Hz

      reference:
      A Deep Learning Framework for Character Motion Synthesis and Editing
  """
  def __init__(self, pca_path=None, padding=True):
    super(MotionEncoder, self).__init__()

    self.mean = nn.Parameter(torch.zeros((1, 73, 1)))
    self.std = nn.Parameter(torch.ones((1, 73, 1)))
    if padding:
      self.conv = nn.Conv1d(73, 256, 25, padding=12)
    else:
      self.conv = nn.Conv1d(73, 256, 25, padding=0)

    self.activation = nn.ReLU()
    self.pooling = nn.MaxPool1d(2)

    if pca_path:
      self.use_pca = True
      data = torch.load(pca_path)
      self.pca_mean = torch.Tensor(data["mean"]).reshape(1, -1, 1)
      self.pca_U = torch.Tensor(data["U"].transpose())
    else:
      self.use_pca = False

  def forward(self, x):
    feature = (x - self.mean)/self.std
    feature = torch.clamp(feature, -5.0, 5.0)
    feature = self.conv(feature)
    feature = self.activation(feature)
    feature = self.pooling(feature)

    if self.use_pca:
      feature = torch.matmul(self.pca_U, feature - self.pca_mean)

    return feature

  def gram_matrix(self, x):
    features = self.forward(x)
    length = x.shape[-1]
    # batched mat mul
    gram_matrix = torch.matmul(features, features.transpose(-1, -2))
    gram_matrix = gram_matrix / length

    return gram_matrix

  def embeding_and_gram_matrix(self, x):
    features = self.forward(x)
    length = x.shape[-1]
    # batched mat mul
    gram_matrix = torch.matmul(features, features.transpose(-1, -2))
    gram_matrix = gram_matrix / length

    return features, gram_matrix

encoder = None

def init_motion_encoder(pca_path=None, padding=True):
  global encoder
  encoder = MotionEncoder(pca_path, padding)
  import pathlib
  dir_path = pathlib.Path(__file__).parent.absolute()
  encoder.load_state_dict(torch.load("%s/motion_encoder.tar" % dir_path))

def process_embedding(feature):
  if encoder is None:
    init_motion_encoder()
  with torch.no_grad():
    t = torch.Tensor(feature)
    emb = encoder(t).cpu().numpy()
  return emb

def process_gram_matrix(feature):
  if encoder is None:
    init_motion_encoder()
  with torch.no_grad():
    t = torch.Tensor(feature)
    G = encoder.gram_matrix(t).cpu().numpy()
  return G

def process_embeding_and_gram_matrix(feature):
  if encoder is None:
    init_motion_encoder()
  with torch.no_grad():
    t = torch.Tensor(feature)
    H, G = encoder.embeding_and_gram_matrix(t)
    H = H.cpu().numpy()
    G = G.cpu().numpy()
  return H, G

if __name__=="__main__":

  encoder_torch = MotionEncoder()
  encoder_torch.load_state_dict(torch.load("motion_encoder.tar"))

  import numpy as np
  a = np.random.rand(1, 73, 120)
  t = torch.Tensor(a)

  G = encoder_torch.gram_matrix(t).data.numpy()

  encoder_pca = MotionEncoder("./walk_pca.tar")
  feature = encoder_pca(t)
  G_pca = encoder_pca.gram_matrix(t).data.numpy()

  from IPython import embed; embed()
