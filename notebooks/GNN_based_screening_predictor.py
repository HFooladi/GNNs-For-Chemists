# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="W0wnn1VctcNz"
# ## Graph Prediction Tasks
# What are the kinds of problems we want to solve on graphs?
#
#
# The tasks fall into roughly three categories:
#
# 1. **Node Classification**: E.g. what is the topic of a paper given a citation network of papers?
# 2. **Link Prediction / Edge Classification**: E.g. are two people in a social network friends?
# 3. **Graph Classification**: E.g. is this protein molecule (represented as a graph) likely going to be effective?
#
# <image src="https://storage.googleapis.com/dm-educational/assets/graph-nets/graph_tasks.png" width="700px">

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 77585, "status": "ok", "timestamp": 1717664517324, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="AlonyZK2AP7t" outputId="bbfb5c1f-8b58-43e9-b2a3-d4ab7495d04b"
#@title Intstall necessary libraries
# !pip install ogb optax

# + id="YWEG8SJMApGo"
#@title Import required libraries
from ogb.graphproppred import GraphPropPredDataset # ogb for data handling
from ogb.graphproppred import Evaluator # ogb for evaluating final prediction

import numpy as np # Ordinary NumPy
from typing import List, Dict # Different types in the notebook

import jax # JAX
import jax.numpy as jnp # JAX NumPy
import optax # Optax for optimization

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4597, "status": "ok", "timestamp": 1717664590562, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="pmEABe_kB7ge" outputId="ce50e8f9-fcc4-49c1-9f08-dc15248b8cb4"
#@title Load the data
dataset = GraphPropPredDataset(name='ogbg-molhiv')

# Get one sample/example from the data
graph, label = dataset[0]

# get some ststs about the data
print("size of the dataset:", len(dataset))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 203, "status": "ok", "timestamp": 1717664594966, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="HWXeAY1BCPLN" outputId="715f1c65-5917-4a61-87c1-f08b346220ba"
print(f'Graph keys are: {graph.keys()}')
print(f'Label for this sample is {label}')

print(graph['num_nodes'])
print(graph['node_feat'].shape)
print(graph['edge_feat'].shape)
print(graph['edge_index'].shape)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 213, "status": "ok", "timestamp": 1717664650325, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="DBCDFQujFD8K" outputId="09cf3176-bdeb-4d22-c71c-481ae85a6d60"
print(graph['node_feat'][:5, :])
print(graph['edge_feat'][:5, :])


# + id="8wjB21SqIJMS"
def convert_edge_index_to_matrix(edge_index: np.ndarray, nb_nodes: int) -> np.ndarray:
  """
  Parameters
  ----------
  edge_index : np.ndarray
    It is [2 x Num_edges] matrix which contains information
    about the sender and receiver node. The first row contains
    sender and the second row contains receiver nodes.
  nb_nods : int
    Number of nodes in the graph

  Returns
  -------
  np.ndarray
    It returns the adjacency matrix of the graph.

  Notes
  -----
  We consider edge from a node to itself (self-edge) in the adjacency matrix.
  So, the diagonal elements are 1.0.
  """
  adj_mat = np.eye(nb_nodes)
  for i in range(edge_index.shape[1]):
    adj_mat[edge_index[0, i], edge_index[1, i]] = 1.0

  return adj_mat / np.sum(adj_mat, axis = -1, keepdims= True)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 201, "status": "ok", "timestamp": 1717664861693, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="_3W-_U8qKUh7" outputId="b4dd8552-785e-4c67-93ab-d44a8d17a1ab"
print(convert_edge_index_to_matrix(graph['edge_index'], graph['num_nodes']))


# + [markdown] id="GKPmcRJNuCTP"
# 1. _Compute messages / update node features_: Create a feature vector $\vec{h}_n$ for each node $n$ (e.g. with an MLP). This is going to be the message that this node will pass to neighboring nodes.
# 2. _Message-passing / aggregate node features_: For each node, calculate a new feature vector $\vec{h}'_n$ based on the messages (features) from the nodes in its neighborhood. In a directed graph, only nodes from incoming edges are counted as neighbors. The image below shows this aggregation step. There are multiple options for aggregation in a GCN, e.g. taking the mean, the sum, the min or max.
#
# <image src="https://storage.googleapis.com/dm-educational/assets/graph-nets/graph_conv.png" width="500px">
#
# *\"A generic overview of a graph convolution operation, highlighting the relevant information for deriving the next-level features for every node in the graph.\"* Image source: Petar Veličković (https://github.com/PetarV-/TikZ)
#
# ## Graph Convolution Network
#
# Let $A$ be the adjacency matrix defining the edges of the graph.
#
# Then we define the degree matrix $D$ as a diagonal matrix with $D_{ii} = \sum_jA_{ij}$ (the degree of node $i$)
#
#
# Now we can normalize $AH$ by dividing it by the node degrees:
# $${D}^{-1}AH$$
#
# To take both the in and out degrees into account, we can use symmetric normalization, which is also what Kipf and Welling proposed in their [paper](https://arxiv.org/abs/1609.02907):
# $$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}H$$
#
# So, the update for each layer can be written as:
#
# $$H^{L+1} = Nonlinearity({D}^{-1}AH^{L}W^{L})$$

# + id="7mWt0Gg3N_53"
@jax.jit
def simple_gnn_layer(weights: np.ndarray, features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
  latent = jnp.matmul(features, weights) # (N x H) * (H x H') -> (N x H')
  latent = jnp.matmul(adj_matrix, latent) # (N x N) * (N x H') -> (N x H')
  latent = jax.nn.relu(latent)
  return latent

@jax.jit
def network(params: List[np.ndarray], features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
  latent = features
  for layer in range(len(params) - 1):
    latent = simple_gnn_layer(params[layer], latent, adj_matrix) # (N x H)

  graph_features = jnp.mean(latent, axis = 0) # (H, )
  logits = jnp.matmul(graph_features, params[-1]) # (1, )
  return logits


# + id="_zxgqqTCVOP6"
@jax.jit
def binary_cross_enrtopy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
  max_val = jnp.clip(logits, 0, None)
  loss = logits - logits * labels + max_val + jnp.log(
      jnp.exp(-max_val) + jnp.exp((-logits - max_val)))
  return jnp.mean(loss)

@jax.jit
def _loss(params, features, adj_matrix, labels) -> np.ndarray:
  logits = network(params, features, adj_matrix)
  return binary_cross_enrtopy(logits, labels)

@jax.jit
def accuracy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
  return jnp.mean((logits > 0) == (labels > 0.5))


# + id="-aGIJ8rbXlqa"
split_idx: Dict = dataset.get_idx_split()

train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
input_dim = graph['node_feat'].shape[1]

def train(hidden_dim: int, nb_layers: int, epochs: int, learning_rate: float) -> List[np.ndarray]:
  params = []
  params.append(np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
  for i in range(nb_layers - 2):
    params.append(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
  params.append(np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim))

  opt = optax.adam(learning_rate = learning_rate)
  opt_state = opt.init(params)

  @jax.jit
  def _step(params, opt_state, features, adj_matrix, labels):
    loss, grads = jax.value_and_grad(_loss)(
        params, features, adj_matrix, labels)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  ep = 0
  step = 0
  while ep < epochs:
    for idx in train_idx:
      graph, label = dataset[idx]
      node_fts = graph['node_feat']
      nb_nodes = graph['num_nodes']
      adj_mat = convert_edge_index_to_matrix(graph['edge_index'], nb_nodes)

      params, opt_state, loss = _step(params, opt_state, node_fts, adj_mat, label)

      if step % 1000 == 0:
        print(f'step: {step} |  loss: {loss}')
      step += 1

    val_preds=[]
    val_labels=[]

    for idx in val_idx:
      graph, label = dataset[idx]
      node_fts = graph['node_feat']
      nb_nodes = graph['num_nodes']
      adj_mat = convert_edge_index_to_matrix(graph['edge_index'], nb_nodes)

      val_preds.append(network(params, node_fts, adj_mat)[0])
      val_labels.append(label)

    val_accuracy = accuracy(jnp.array(val_preds), jnp.array(val_labels))
    print(f'epochs: {ep} | validation accuracy: {val_accuracy} ')

    ep += 1

  return params


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 109469, "status": "ok", "timestamp": 1717664977361, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="t73a64Zpbtem" outputId="a46f3afb-9d39-4f44-9e46-a5203dd3ddce"
trained_params = train(hidden_dim=32, nb_layers=2, epochs=2, learning_rate=0.001)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6263, "status": "ok", "timestamp": 1717665182801, "user": {"displayName": "Hosein Fooladi", "userId": "05481041717235244396"}, "user_tz": -120} id="AFfhIh9Pigxi" outputId="2ca198d5-d788-404a-c899-ef997598f8f4"
#@title Evaluating the model performance on Test data (ROC AUC)
def sigmoid(x):
  return 1./(1. + np.exp(-x))

test_preds=[]
test_labels=[]

for idx in test_idx:
  graph, label = dataset[idx]
  node_fts = graph['node_feat']
  nb_nodes = graph['num_nodes']
  adj_mat = convert_edge_index_to_matrix(graph['edge_index'], nb_nodes)

  test_preds.append(network(trained_params, node_fts, adj_mat)[0])
  test_labels.append(label)

test_accuracy = accuracy(jnp.array(test_preds), jnp.array(test_labels))
print(f'Test accuracy: {test_accuracy} ')

evaluator = Evaluator(name = "ogbg-molhiv")
input_dict = {"y_true": np.array(test_labels), "y_pred": sigmoid(np.array(test_preds)).reshape(-1, 1)}
result_dict = evaluator.eval(input_dict)
print(f'ROC AUC: {result_dict["rocauc"]} ')


# + [markdown] id="seJnrN2GXQ8P"
# ### Exercise
#
# 1. Change the hyperparameters (number of layesrs, size of each layer) and evaluate the performance? What is the effect of eahc change? What is the best results you can achieve by changing hyperparameters?
#
# 2. Can you implement exactly the same GCN, but with adding skip connection into the architecture?
#
# 3. What are the different approaches for increasing the accuracy of the prediction?

# + id="N2J_QzCNjGZT"

