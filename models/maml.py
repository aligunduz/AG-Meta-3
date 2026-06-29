from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d


def _param_name_to_key(name):
  return name.replace('.', '__')


def _default_multi_anchor_args(multi_anchor_args=None):
  args = {
    'n_anchor': 3,
    'tau': 1.0,
    'detach_routing_weights': True,
    'init_noise_std': 0.01,
    'log_stats': True,
    'log_anchor_diagnostics': True,
    'anchor_log_interval': 20,
    'save_anchor_snapshots': True,
    'snapshot_interval': 20,
  }
  if multi_anchor_args is not None:
    args.update(multi_anchor_args)
  args['n_anchor'] = int(args['n_anchor'])
  args['tau'] = float(args['tau'])
  args['init_noise_std'] = float(args['init_noise_std'])
  if args['n_anchor'] < 1:
    raise ValueError('multi_anchor.n_anchor must be >= 1')
  if args['tau'] <= 0:
    raise ValueError('multi_anchor.tau must be > 0')
  return args


def _vector_norm(x, dim=None):
  if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'vector_norm'):
    return torch.linalg.vector_norm(x, dim=dim)
  return torch.norm(x, p=2, dim=dim)


def _symmetric_eigh(x):
  if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigh'):
    return torch.linalg.eigh(x)
  return torch.symeig(x, eigenvectors=True)


class _AnchorParamSet(nn.Module):
  def __init__(self, named_params, init_noise_std):
    super(_AnchorParamSet, self).__init__()
    self.params = nn.ParameterDict()
    self.names = []

    for name, param in named_params:
      key = _param_name_to_key(name)
      init_value = param.detach().clone()
      if init_noise_std > 0:
        init_value = init_value + torch.randn_like(init_value) * init_noise_std
      self.params[key] = nn.Parameter(init_value)
      self.register_buffer('init__' + key, init_value.detach().clone())
      self.names.append(name)

  def get_param(self, name):
    return self.params[_param_name_to_key(name)]

  def get_init_param(self, name):
    return getattr(self, 'init__' + _param_name_to_key(name))


def make(
        enc_name,
        enc_args,
        clf_name,
        clf_args,
        use_multi_anchor=False,
        multi_anchor_args=None):
  """
  Initializes a random meta model.

  Args:
    enc_name (str): name of the encoder (e.g., 'resnet12').
    enc_args (dict): arguments for the encoder.
    clf_name (str): name of the classifier (e.g., 'meta-nn').
    clf_args (dict): arguments for the classifier.

  Returns:
    model (MAML): a meta classifier with a random encoder.
  """
  enc = encoders.make(enc_name, **enc_args)
  clf_args['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(clf_name, **clf_args)
  model = MAML(
    enc, clf,
    use_multi_anchor=use_multi_anchor,
    multi_anchor_args=multi_anchor_args)
  return model


def load(
        ckpt,
        load_clf=False,
        clf_name=None,
        clf_args=None,
        use_multi_anchor=None,
        multi_anchor_args=None):
  """
  Initializes a meta model with a pre-trained encoder.

  Args:
    ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
    load_clf (bool, optional): if True, loads a pre-trained classifier.
      Default: False (in which case the classifier is randomly initialized)
    clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
    clf_args (dict, optional): arguments for the classifier.
    (The last two arguments are ignored if load_clf=True.)

  Returns:
    model (MAML): a meta model with a pre-trained encoder.
  """
  enc = encoders.load(ckpt)
  if load_clf:
    clf = classifiers.load(ckpt)
  else:
    if clf_name is None and clf_args is None:
      clf = classifiers.make(ckpt['classifier'], **ckpt['classifier_args'])
    else:
      clf_args['in_dim'] = enc.get_out_dim()
      clf = classifiers.make(clf_name, **clf_args)
  ckpt_config = ckpt.get('config') or {}
  if use_multi_anchor is None:
    use_multi_anchor = ckpt_config.get(
      'use_multi_anchor', 'multi_anchor_state_dict' in ckpt)

  ckpt_multi_anchor_args = dict(ckpt_config.get('multi_anchor') or {})
  if multi_anchor_args is not None:
    ckpt_multi_anchor_args.update(multi_anchor_args)

  model = MAML(
    enc, clf,
    use_multi_anchor=use_multi_anchor,
    multi_anchor_args=ckpt_multi_anchor_args)
  if 'gradient_transport_state_dict' in ckpt:
    model.gradient_transport_logits.load_state_dict(
      ckpt['gradient_transport_state_dict'])
  if model.use_multi_anchor and 'multi_anchor_state_dict' in ckpt:
    model.load_multi_anchor_state_dict(ckpt['multi_anchor_state_dict'])
  return model


class MAML(Module):
  def __init__(
          self,
          encoder,
          classifier,
          use_multi_anchor=False,
          multi_anchor_args=None):
    super(MAML, self).__init__()
    self.encoder = encoder
    self.classifier = classifier
    self.gradient_transport_logits = nn.ParameterDict()
    self.use_multi_anchor = bool(use_multi_anchor)
    self.multi_anchor_args = _default_multi_anchor_args(
      multi_anchor_args if self.use_multi_anchor else None)
    self.multi_anchor_bank = nn.ModuleList()
    self.multi_anchor_param_names = []
    self.last_multi_anchor_stats = {}
    device = next(self.parameters()).device

    for name, _ in self.encoder.named_parameters():
      key = 'encoder__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(
        torch.tensor(4.0, device=device))

    for name, _ in self.classifier.named_parameters():
      key = 'classifier__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(
        torch.tensor(4.0, device=device))

    if self.use_multi_anchor:
      self._init_multi_anchor_bank()

  def reset_classifier(self):
    self.classifier.reset_parameters()

  def _base_named_parameters(self):
    params = OrderedDict()
    for name, param in self.encoder.named_parameters():
      params['encoder.' + name] = param
    for name, param in self.classifier.named_parameters():
      params['classifier.' + name] = param
    return params

  def _is_fast_weight_name(self, name):
    if not (name.startswith('encoder.') or name.startswith('classifier.')):
      return False
    return 'temp' not in name

  def _is_adaptable_param(self, name, param, inner_args):
    if not param.requires_grad:
      return False
    blocked = inner_args['frozen'] + ['temp']
    return not any(s in name for s in blocked)

  def _get_base_adaptation_params(self, inner_args):
    params = self._base_named_parameters()
    for name in list(params.keys()):
      if not self._is_adaptable_param(name, params[name], inner_args):
        params.pop(name)
    return params

  def _init_multi_anchor_bank(self):
    base_params = [
      (name, param)
      for name, param in self._base_named_parameters().items()
      if self._is_fast_weight_name(name)
    ]
    self.multi_anchor_param_names = [name for name, _ in base_params]
    for _ in range(self.multi_anchor_args['n_anchor']):
      self.multi_anchor_bank.append(
        _AnchorParamSet(
          base_params,
          self.multi_anchor_args['init_noise_std']))

  def load_multi_anchor_state_dict(self, state_dict):
    if not self.use_multi_anchor:
      return [], list(state_dict.keys())
    missing, unexpected = self.multi_anchor_bank.load_state_dict(
      state_dict, strict=False)
    return missing, unexpected

  def get_multi_anchor_state_dict(self):
    if not self.use_multi_anchor:
      return OrderedDict()
    return self.multi_anchor_bank.state_dict()

  def get_gradient_transport_gates(self):
    out = {}
    for key, logit in self.gradient_transport_logits.items():
      out[key] = torch.sigmoid(logit).detach().item()
    return out

  def get_last_multi_anchor_stats(self):
    return dict(self.last_multi_anchor_stats)

  def _reset_episodic_batch_norm(self, episode):
    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

  def _capture_batch_norm_state(self):
    state = []
    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.track_running_stats:
        buffers = []
        for name, buffer in m.named_buffers(recurse=False):
          buffers.append((name, buffer.detach().clone()))
        state.append((m, buffers))
    return state

  def _restore_batch_norm_state(self, state):
    for module, buffers in state:
      for name, value in buffers:
        getattr(module, name).copy_(value)

  def _get_anchor_params(self, anchor_idx, inner_args=None):
    anchor = self.multi_anchor_bank[anchor_idx]
    base_params = self._base_named_parameters()
    params = OrderedDict()
    for name in self.multi_anchor_param_names:
      if inner_args is not None and \
          not self._is_adaptable_param(name, base_params[name], inner_args):
        continue
      params[name] = anchor.get_param(name)
    return params

  def _compute_anchor_routing(self, x, y, episode, inner_args, meta_train):
    self._reset_episodic_batch_norm(episode)
    clean_bn_state = self._capture_batch_norm_state()
    losses = []
    grad_enabled = (
      meta_train and
      not bool(self.multi_anchor_args['detach_routing_weights']))

    with torch.set_grad_enabled(grad_enabled):
      for anchor_idx in range(len(self.multi_anchor_bank)):
        self._restore_batch_norm_state(clean_bn_state)
        anchor_params = self._get_anchor_params(anchor_idx, inner_args)
        logits = self._inner_forward(x, anchor_params, episode)
        losses.append(F.cross_entropy(logits, y))

    self._restore_batch_norm_state(clean_bn_state)
    support_losses = torch.stack(losses)
    routing_weights = F.softmax(
      -support_losses / self.multi_anchor_args['tau'], dim=0)
    if self.multi_anchor_args['detach_routing_weights']:
      routing_weights = routing_weights.detach()

    return routing_weights, support_losses.detach()

  def _mix_anchor_params(self, routing_weights, inner_args, meta_train):
    base_params = self._base_named_parameters()
    params = OrderedDict()
    with torch.enable_grad():
      for name in self.multi_anchor_param_names:
        if not self._is_adaptable_param(name, base_params[name], inner_args):
          continue

        mixed_param = None
        for anchor_idx, weight in enumerate(routing_weights):
          anchor_param = self.multi_anchor_bank[anchor_idx].get_param(name)
          weighted_param = weight * anchor_param
          if mixed_param is None:
            mixed_param = weighted_param
          else:
            mixed_param = mixed_param + weighted_param

        if meta_train:
          if not mixed_param.requires_grad:
            mixed_param = mixed_param.clone().requires_grad_(True)
        else:
          mixed_param = mixed_param.detach().requires_grad_(True)
        params[name] = mixed_param
    return params

  def _routing_stats(self, routing_weights, support_losses):
    stats = {}
    weights = routing_weights.detach().float()
    losses = support_losses.detach().float()
    entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-12)))
    stats['multi_anchor/routing_entropy'] = entropy
    stats['multi_anchor/max_weight_mean'] = torch.max(weights)
    for anchor_idx in range(weights.numel()):
      stats[
        'multi_anchor/anchor_{}_weight_mean'.format(anchor_idx)
      ] = weights[anchor_idx]
      stats[
        'multi_anchor/anchor_{}_support_loss'.format(anchor_idx)
      ] = losses[anchor_idx]
    return stats

  def _set_last_multi_anchor_stats(self, stats_per_episode):
    if not stats_per_episode:
      self.last_multi_anchor_stats = {}
      return

    stats = {}
    keys = sorted(stats_per_episode[0].keys())
    for key in keys:
      values = [ep_stats[key] for ep_stats in stats_per_episode]
      stats[key] = torch.stack(values).mean().item()
    self.last_multi_anchor_stats = stats

  def _anchor_vectors(self):
    vectors, init_vectors = [], []
    for anchor in self.multi_anchor_bank:
      parts, init_parts = [], []
      for name in self.multi_anchor_param_names:
        parts.append(anchor.get_param(name).detach().reshape(-1).float().cpu())
        init_parts.append(
          anchor.get_init_param(name).detach().reshape(-1).float().cpu())
      vectors.append(torch.cat(parts))
      init_vectors.append(torch.cat(init_parts))
    return torch.stack(vectors), torch.stack(init_vectors)

  def _anchor_pca(self, vectors):
    if vectors.size(0) == 1:
      return torch.zeros(1, 2)

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    gram = centered.mm(centered.t())
    eigvals, eigvecs = _symmetric_eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.)
    eigvecs = eigvecs[:, order]
    n_comp = min(2, eigvals.numel())
    coords = eigvecs[:, :n_comp] * eigvals[:n_comp].sqrt().view(1, -1)
    if n_comp < 2:
      coords = torch.cat([coords, torch.zeros(coords.size(0), 1)], dim=1)
    return coords

  def get_multi_anchor_diagnostics(self):
    if not self.use_multi_anchor:
      return None

    vectors, init_vectors = self._anchor_vectors()
    metrics = {}
    norms = _vector_norm(vectors, dim=1)
    delta_norms = _vector_norm(vectors - init_vectors, dim=1)
    for anchor_idx in range(vectors.size(0)):
      metrics[
        'multi_anchor/anchor_{}_norm'.format(anchor_idx)
      ] = norms[anchor_idx].item()
      metrics[
        'multi_anchor/anchor_{}_delta_from_init_norm'.format(anchor_idx)
      ] = delta_norms[anchor_idx].item()

    diffs = vectors.unsqueeze(1) - vectors.unsqueeze(0)
    distance = torch.sqrt(torch.clamp(torch.sum(diffs * diffs, dim=2), min=0.))
    normalized = F.normalize(vectors, dim=1, eps=1e-12)
    cosine = normalized.mm(normalized.t())
    pair_distances, pair_cosines = [], []
    for i in range(vectors.size(0)):
      for j in range(i + 1, vectors.size(0)):
        dist = distance[i, j].item()
        cos = cosine[i, j].item()
        metrics['multi_anchor/pairwise_distance_{}_{}'.format(i, j)] = dist
        metrics['multi_anchor/pairwise_cosine_{}_{}'.format(i, j)] = cos
        pair_distances.append(dist)
        pair_cosines.append(cos)
    metrics['multi_anchor/mean_pairwise_distance'] = (
      sum(pair_distances) / len(pair_distances)
      if pair_distances else 0.)
    metrics['multi_anchor/mean_pairwise_cosine'] = (
      sum(pair_cosines) / len(pair_cosines)
      if pair_cosines else 1.)

    pca_coords = self._anchor_pca(vectors)
    return {
      'metrics': metrics,
      'anchor_norms': norms.tolist(),
      'anchor_delta_from_init_norms': delta_norms.tolist(),
      'pairwise_distance_matrix': distance.tolist(),
      'pairwise_cosine_matrix': cosine.tolist(),
      'pca_coords': pca_coords.tolist(),
    }

  def save_multi_anchor_snapshot(self, path, epoch, diagnostics=None):
    if not self.use_multi_anchor:
      return
    if diagnostics is None:
      diagnostics = self.get_multi_anchor_diagnostics()
    payload = {
      'epoch': epoch,
      'multi_anchor_args': self.multi_anchor_args,
      'anchor_state_dict': self.get_multi_anchor_state_dict(),
      'anchor_norms': diagnostics['anchor_norms'],
      'anchor_delta_from_init_norms':
        diagnostics['anchor_delta_from_init_norms'],
      'pairwise_distance_matrix': diagnostics['pairwise_distance_matrix'],
      'pairwise_cosine_matrix': diagnostics['pairwise_cosine_matrix'],
      'pca_coords': diagnostics['pca_coords'],
    }
    torch.save(payload, path)

  def _inner_forward(self, x, params, episode):
    feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits

  def _inner_iter(
          self,
          x,
          y,
          params,
          mom_buffer,
          episode,
          inner_args,
          detach,
          use_gradient_transport=False):
    """
    Performs one inner-loop iteration of MAML, optionally applying a learned
    scalar gradient transport gate per parameter tensor.
    """
    with torch.enable_grad():
      logits = self._inner_forward(x, params, episode)
      loss = F.cross_entropy(logits, y)
      grads = autograd.grad(loss, params.values(),
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)

      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          if use_gradient_transport:
            gate_key = name.replace('.', '__')
            gate = torch.sigmoid(self.gradient_transport_logits[gate_key])
            grad = gate * grad
          updated_param = param - lr * grad
        if detach:
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _adapt(
          self,
          x,
          y,
          params,
          episode,
          inner_args,
          meta_train,
          use_gradient_transport=False):
    """
    Performs inner-loop adaptation in MAML.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)

    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    self._reset_episodic_batch_norm(episode)

    def _inner_iter_cp(episode, *state):
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach,
        use_gradient_transport=use_gradient_transport)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._inner_iter(
          x, y, params, mom_buffer, episode, inner_args, not meta_train,
          use_gradient_transport=use_gradient_transport)

    return params

  def forward(
          self,
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train,
          use_gradient_transport=False):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.

    Returns:
      logits (float tensor, [n_episode, n_way * n_query, n_way]): query logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    base_params = self._get_base_adaptation_params(inner_args)

    logits = []
    multi_anchor_stats = []
    for ep in range(x_shot.size(0)):
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()

      if self.use_multi_anchor:
        routing_weights, support_losses = self._compute_anchor_routing(
          x_shot[ep], y_shot[ep], ep, inner_args, meta_train)
        params = self._mix_anchor_params(
          routing_weights, inner_args, meta_train)
        multi_anchor_stats.append(
          self._routing_stats(routing_weights, support_losses))
      else:
        params = base_params

      updated_params = self._adapt(
        x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train,
        use_gradient_transport=use_gradient_transport)

      with torch.set_grad_enabled(meta_train):
        self.eval()
        logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
      logits.append(logits_ep)

    self.train(meta_train)
    self._set_last_multi_anchor_stats(multi_anchor_stats)
    return torch.stack(logits)
