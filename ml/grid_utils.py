# from Thiago Hersan - thiagohersan
# https://gist.github.com/thiagohersan/e34aa4648dbbd60483ade095c3b64734

import lap
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist

def get_grid_xy(data, grid_dims=None, px_dims=None):
  # normalize features
  data2d = data - data.min(axis=0)
  data2d /= data2d.max(axis=0)
  data2d = data2d[:, :2]

  if grid_dims == None:
    side = int(data.shape[0]**0.5) + 1
    grid_dims = (side, side)
  elif type(grid_dims) == int:
    grid_dims = (grid_dims, grid_dims)

  xv, yv = np.meshgrid(np.linspace(0, 1, grid_dims[0]), np.linspace(0, 1, grid_dims[1]))
  grid = np.dstack((xv, yv)).reshape(-1, 2)

  # pairwise cost/distance between all combinations of images and grid points
  _cost = cdist(grid, data2d, 'sqeuclidean')

  # exagerate the cost of moving
  scale_f = 10000000.
  cost = _cost * (scale_f / _cost.max())

  # lap does row/col assignment while minimizing the distance that each point is moved
  min_cost, row_assigns, col_assigns = lap.lapjv(np.copy(cost).astype(int), extend_cost=True)
  grid_jv = grid[col_assigns]

  if px_dims is not None:
    if type(px_dims) == int or type(px_dims) == float:
      px_dims = (px_dims, px_dims)
    grid_jv[:, 0] *= px_dims[0]
    grid_jv[:, 1] *= px_dims[1]
  
  return grid_jv


def plot_moves(data, grid, px_dims=None):
  data2d = data - data.min(axis=0)
  data2d /= data2d.max(axis=0)
  data2d = data2d[:, :2]

  if px_dims is not None:
    if type(px_dims) == int or type(px_dims) == float:
      px_dims = (px_dims, px_dims)
    grid = grid.copy()
    grid[:, 0] /= px_dims[0]
    grid[:, 1] /= px_dims[1]

  plt.figure(figsize=(18, 18))
  for start, end in zip(data2d, grid):
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_length=0.003, head_width=0.003, color=(0,0,0,.15))
  plt.show()