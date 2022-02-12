def visualize_segments(dataset, idx = 0):
    """Plots the boundaries of segments from a torch Dataset object.
      Input: dataset with the following attr: image, segments, idx img to plot
      Requirements: torch, skimage(img_as_float)

    Args:
      dataset: 
      idx:  (Default value = 0)

    Returns:

    """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  fig, ax = plt.subplots(1, figsize=(10, 10))
  plt.imshow(
      mark_boundaries(
          dataset[idx]["image"], 
          dataset[idx]["segments"]
          )
      )
  plt.show()

def visualize_rag_graph(dataset, idx = 0):
    """Plot rag mean-color graph on top of original picture

    Args:
      Requirements: matplotlib.pyplot as plt
    numpy as np
    torch
    skimage.future.graph
      dataset: 
      idx:  (Default value = 0)

    Returns:

    """
  img = np.array(dataset[idx]["image"])
  segments = np.array(dataset[idx]["segments"])
  g = graph.rag_mean_color(img, segments)

  #Create plot
  fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

  #Create dummy mask of all ones for white background (bottom img)
  dummy = np.ones_like(img, dtype = np.float64)

  #Specify the fraction of the plot area that will be used to draw the colorbar
  #Set title and plot
  ax[0].set_title('RAG drawn with default settings')
  lc = graph.show_rag(segments, g, img, ax=ax[0], border_color ="white")
  fig.colorbar(lc, fraction=0.1, ax=ax[0])

  #Plot again on white background
  ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
  lc = graph.show_rag(segments, g, dummy,
                      img_cmap='gray', 
                      edge_cmap='viridis', 
                      ax=ax[1], border_color ="white")
  ax[1].imshow(dummy)
  fig.colorbar(lc, fraction=0.1, ax=ax[1])

  #Set axes off
  for a in ax:
      a.axis('off')

  plt.tight_layout()
  plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    """

    Args:
      h: 
      color: 
      epoch:  (Default value = None)
      loss:  (Default value = None)

    Returns:

    """
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()