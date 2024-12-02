import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

def generate_array_data_for_recursive_buckets(y_real, y_generated, uncertainty_scores, x_sample, method='fd'):
  data = []

  for i in range(len(x_sample)):
    observation = []
    observation.extend(x_sample[i])
    observation.extend(y_real[i])
    observation.extend(y_generated[i])
    if method == 'fd':
      observation.append(uncertainty_scores[i])
    else:
      observation.extend(uncertainty_scores[i])
    data.append(observation)

  return data

def create_recursive_buckets(data, num_buckets = 3, min_bucket = 10, max_bucket = 50):
  x = torch.tensor(data, dtype=torch.float32)
  x_min_1 = torch.min(x, dim=0).values
  x_max_1 = torch.max(x, dim=0).values
  buckets_0 = torch.linspace(x_min_1[0], x_max_1[0], num_buckets)
  buckets_1 = torch.linspace(x_min_1[1], x_max_1[1], num_buckets)
  buckets_2 = torch.linspace(x_min_1[2], x_max_1[2], num_buckets)
  #List to store the buckets of data, we can extract info of this buckets later
  cubes = []
  #First Bucket Unpopulated
  FBU = False
  for curr_bucket_0 in range(num_buckets - 1):
    for curr_bucket_1 in range(num_buckets - 1):
      for curr_bucket_2 in range(num_buckets - 1):
          lower_bound_0 = buckets_0[curr_bucket_0]
          upper_bound_0 = buckets_0[curr_bucket_0 + 1]
          lower_bound_1 = buckets_1[curr_bucket_1]
          upper_bound_1 = buckets_1[curr_bucket_1 + 1]
          lower_bound_2 = buckets_2[curr_bucket_2]
          upper_bound_2 = buckets_2[curr_bucket_2 + 1]

          # Save the points that falls into the bucket
          data_cube = x[(x[:, 0] > lower_bound_0) & (x[:, 0] <= upper_bound_0) &
                        (x[:, 1] > lower_bound_1) & (x[:, 1] <= upper_bound_1) &
                        (x[:, 2] > lower_bound_2) & (x[:, 2] <= upper_bound_2)]

          # If the bucket has some points, save it
          if FBU and data_cube.shape[0] < min_bucket and data_cube.shape[0] > 0:
            cubes[-1] = torch.cat((cubes[-1], data_cube), dim=0)
          elif FBU and data_cube.shape[0] > 0:
            cubes[-1] = torch.cat((cubes[-1], data_cube), dim=0)
            FBU = False
          elif data_cube.shape[0] < min_bucket and data_cube.shape[0] > 0:
            try:
              cubes[-1] = torch.cat((cubes[-1], data_cube), dim=0)
            except:
              FBU = True
              cubes.append(data_cube)
          elif data_cube.shape[0] > 0:
            cubes.append(data_cube)

  # Recursively, divide the buckets in sub buckets, so every bucket will have 0-x points
  for i in cubes:
    if i.shape[0] > max_bucket:
      cubes_i = create_recursive_buckets(i, 3)
      cubes.extend(cubes_i)

  for i in range(len(cubes) -1, -1, -1):
    if cubes[i].shape[0] == 0 or cubes[i].shape[0] > max_bucket:
      cubes.pop(i)

  return cubes

def firstTwo(numero):
    # Convertimos el número a cadena, removemos el punto decimal si es necesario
    str_num = str(numero).replace('.', '')

    # Tomamos los primeros dos caracteres y los convertimos de nuevo a entero
    return str_num[0] + '.' + str_num[1]

def get_limits(tags):
  min_value = 10000
  max_value = 0
  for i in range(5):
    min_value = min(min_value, min(tags[i]))
    max_value = max(max_value, max(tags[i]))

  return min_value, max_value

#This function creates the 2D plots with the quantil tracks method of plotting
def plot_2D_tracks(tags, data, title, prefix, color='inferno'):
  nombres = [prefix + ' RichDLLe', prefix + ' RichDLLk', prefix + ' RichDLLmu', prefix + ' RichDLLp', prefix + ' RichDLLbt']
  tags = np.array(tags)

  # Quantiles
  bin_edges = np.quantile(data[:,7], np.linspace(0, 1, 5))

  min_value, max_value = get_limits(tags)

  for rich in range (5):
    #fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 fila, 4 columnas
    fig = plt.figure(figsize=(21, 5))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    norm = mcolors.Normalize(min_value, max_value)
    scatter = axs[-1].scatter(tags[5], tags[6], c=tags[rich], marker='.', cmap=color, norm=norm, alpha=1)
    #plt.colorbar(scatter, ax=axs[-1], label=title)
    fig.colorbar(scatter, cax=fig.add_subplot(gs[4]), label=title)
    axs[-1].clear()
    for bin in range(4):
      observations = tags[:,(tags[7,:] >= bin_edges[bin]) & (tags[7,:] < bin_edges[bin+1])]
      scatter = axs[bin].scatter(observations[5], observations[6], c=observations[rich], marker='.', cmap=color, norm=norm, alpha=0.5)
      axs[bin].set_xlabel('momentum P')
      axs[bin].set_ylabel('pseudorapidity η')

      '''
      axs[bin].set_xscale('log')
      xticks = np.logspace(np.log10(min(data[:,5])), np.log10(max(data[:,5])), 5)
      axs[bin].set_xticks(xticks)
      axs[bin].set_xticklabels([f'{firstTwo(tick)}x$10^{{{int(np.log10(tick))}}}$' for tick in xticks])
      '''

      axs[bin].set_xticks(np.linspace(min(data[:,5]),max(data[:,5]),5))
      axs[bin].set_yticks(np.linspace(min(data[:,6]),max(data[:,6]),5))
      axs[bin].set_xlim(min(data[:,5]), max(data[:,5]))
      axs[bin].set_ylim(min(data[:,6]), max(data[:,6]))
      axs[bin].grid(True)
      axs[bin].set_title(f'tracks [{round(bin_edges[bin], 3)}, {round(bin_edges[bin+1], 3)}]')

    fig.suptitle(nombres[rich])

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

#This function gets the distances from the cubes to plot it
def get_distance_tags(data, distance_method):
  '''
  r_e = np.maximum(i[:, 3], 1e-10)
  r_k = np.maximum(i[:, 4], 1e-10)
  r_mu = np.maximum(i[:, 5], 1e-10)
  r_p = np.maximum(i[:, 6], 1e-10)
  r_bt = np.maximum(i[:, 7], 1e-10)
  g_e = np.maximum(i[:, 8], 1e-10)
  g_k = np.maximum(i[:, 9], 1e-10)
  g_mu = np.maximum(i[:, 10], 1e-10)
  g_p = np.maximum(i[:, 11], 1e-10)
  g_bt = np.maximum(i[:, 12], 1e-10)
  '''
  electron, kaon, muon, proton, bt = [], [], [], [], []
  for i in data:
    min_val = min(min(i[:, 3]), min(i[:, 8]))
    if min_val < 0:
        r_e = i[:, 3] + abs(min_val) + 1e-10
        g_e = i[:, 8] + abs(min_val) + 1e-10
    else:
        r_e = i[:, 3]
        g_e = i[:, 8]
    min_val = min(min(i[:, 4]), min(i[:, 9]))
    if min_val < 0:
        r_k = i[:, 4] + abs(min_val) + 1e-10
        g_k = i[:, 9] + abs(min_val) + 1e-10
    else:
        r_k = i[:, 4]
        g_k = i[:, 9]
    min_val = min(min(i[:, 5]), min(i[:, 10]))
    if min_val < 0:
        r_mu = i[:, 5] + abs(min_val) + 1e-10
        g_mu = i[:, 10] + abs(min_val) + 1e-10
    else:
        r_mu = i[:, 5]
        g_mu = i[:, 10]
    min_val = min(min(i[:, 6]), min(i[:, 11]))
    if min_val < 0:
        r_p = i[:, 6] + abs(min_val) + 1e-10
        g_p = i[:, 11] + abs(min_val) + 1e-10
    else:
        r_p = i[:, 6]
        g_p = i[:, 11]
    min_val = min(min(i[:, 7]), min(i[:, 12]))
    if min_val < 0:
        r_bt = i[:, 7] + abs(min_val) + 1e-10
        g_bt = i[:, 12] + abs(min_val) + 1e-10
    else:
        r_bt = i[:, 7]
        g_bt = i[:, 12]

    electron.append(distance_method(r_e, g_e))
    kaon.append(distance_method(r_k, g_k))
    muon.append(distance_method(r_mu, g_mu))
    proton.append(distance_method(r_p, g_p))
    bt.append(distance_method(r_bt,g_bt))
  pDistance = [torch.mean(i[:, 0]) for i in data]
  etaDistance = [torch.mean(i[:, 1]) for i in data]
  tracksDistance = [torch.mean(i[:, 2]) for i in data]

  return [electron, kaon, muon, proton, bt, pDistance, etaDistance, tracksDistance]

#This function gets the uncertainties from the cubes to plot it
def get_uncertainties_tags(data, Umethod='fd'):
  if Umethod == 'fd':
    uncertainties = [[[i[-1].item(),i[-1].item(),i[-1].item(),i[-1].item(),i[-1].item()] for i in cube] for cube in data]
    uncertainties = [torch.tensor(i) for i in uncertainties]
  else:
    uncertainties = [torch.stack([i[-5:] for i in cube]) for cube in data]

  electron = [torch.mean(i[:, 0]) for i in uncertainties]
  kaon = [torch.mean(i[:, 1]) for i in uncertainties]
  muon = [torch.mean(i[:, 2]) for i in uncertainties]
  proton = [torch.mean(i[:, 3]) for i in uncertainties]
  bt = [torch.mean(i[:, 4]) for i in uncertainties]
  pDistance = [torch.mean(i[:, 0]) for i in data]
  etaDistance = [torch.mean(i[:, 1]) for i in data]
  tracksDistance = [torch.mean(i[:, 2]) for i in data]

  return [electron, kaon, muon, proton, bt, pDistance, etaDistance, tracksDistance]

def get_generated_std_tags(data):
  electron = [torch.std(i[:, 8]) for i in data]
  kaon = [torch.std(i[:, 9]) for i in data]
  muon = [torch.std(i[:, 10]) for i in data]
  proton = [torch.std(i[:, 11]) for i in data]
  bt = [torch.std(i[:, 12]) for i in data]
  pDistance = [torch.mean(i[:, 0]) for i in data]
  etaDistance = [torch.mean(i[:, 1]) for i in data]
  tracksDistance = [torch.mean(i[:, 2]) for i in data]

  return [electron, kaon, muon, proton, bt, pDistance, etaDistance, tracksDistance]

def get_std_tags(data):
  electron = [torch.std(i[:, 3]) for i in data]
  kaon = [torch.std(i[:, 4]) for i in data]
  muon = [torch.std(i[:, 5]) for i in data]
  proton = [torch.std(i[:, 6]) for i in data]
  bt = [torch.std(i[:, 7]) for i in data]
  pDistance = [torch.mean(i[:, 0]) for i in data]
  etaDistance = [torch.mean(i[:, 1]) for i in data]
  tracksDistance = [torch.mean(i[:, 2]) for i in data]

  return [electron, kaon, muon, proton, bt, pDistance, etaDistance, tracksDistance]

def get_SDAE_tags(data):
  electron = [torch.abs(torch.std(i[:, 3]) - torch.std(i[:, 8])) for i in data]
  kaon = [torch.abs(torch.std(i[:, 4]) - torch.std(i[:, 9])) for i in data]
  muon = [torch.abs(torch.std(i[:, 5]) - torch.std(i[:, 10])) for i in data]
  proton = [torch.abs(torch.std(i[:, 6]) - torch.std(i[:, 11])) for i in data]
  bt = [torch.abs(torch.std(i[:, 7]) - torch.std(i[:, 12])) for i in data]
  pDistance = [torch.mean(i[:, 0]) for i in data]
  etaDistance = [torch.mean(i[:, 1]) for i in data]
  tracksDistance = [torch.mean(i[:, 2]) for i in data]

  return [electron, kaon, muon, proton, bt, pDistance, etaDistance, tracksDistance]
