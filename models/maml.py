from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d


def make(enc_name, enc_args, clf_name, clf_args):
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
  model = MAML(enc, clf)
  return model


def load(ckpt, load_clf=False, clf_name=None, clf_args=None):
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
  model = MAML(enc, clf)
  if 'gradient_transport_state_dict' in ckpt:
      model.gradient_transport_logits.load_state_dict(
          ckpt['gradient_transport_state_dict']
      )
  if 'task_gate_gamma_state_dict' in ckpt:
      model.task_gate_gammas.load_state_dict(
          ckpt['task_gate_gamma_state_dict']
      )
  return model


class MAML(Module):
  def __init__(self, encoder, classifier):
    super(MAML, self).__init__()
    self.encoder = encoder
    self.classifier = classifier
    # Her katman için 1 öğrenilebilir gate logit'i
    self.gradient_transport_logits = nn.ParameterDict()
    self.task_gate_gammas = nn.ParameterDict()

    # Encoder katmanları
    for name, _ in self.encoder.named_parameters():
      key = 'encoder__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(torch.tensor(4.0))
      self.task_gate_gammas[key] = nn.Parameter(torch.tensor(0.0))

    # Classifier katmanları
    for name, _ in self.classifier.named_parameters():
      key = 'classifier__' + name.replace('.', '__')
      self.gradient_transport_logits[key] = nn.Parameter(torch.tensor(4.0))
      self.task_gate_gammas[key] = nn.Parameter(torch.tensor(0.0))

    self._task_gate_stats = None

  def reset_classifier(self):
    self.classifier.reset_parameters()

  def get_gradient_transport_gates(self):
      out = {}
      for key, logit in self.gradient_transport_logits.items():
          out[key] = torch.sigmoid(logit).detach().item()
      return out

  def get_task_gate_gammas(self):
      out = {}
      for key, gamma in self.task_gate_gammas.items():
          out[key] = gamma.detach().item()
      return out

  def task_gate_gamma_l2(self):
      if len(self.task_gate_gammas) == 0:
          device = next(self.parameters()).device
          return torch.tensor(0., device=device)
      penalties = [gamma.pow(2) for gamma in self.task_gate_gammas.values()]
      return torch.stack(penalties).mean()

  def reset_task_gate_stats(self):
      self._task_gate_stats = {
          'effective_gate_sum': 0.0,
          'effective_gate_min': None,
          'effective_gate_max': None,
          'task_signal_sum': 0.0,
          'count': 0,
      }

  def get_task_gate_stats(self):
      stats = self._task_gate_stats
      if stats is None or stats['count'] == 0:
          return None

      count = stats['count']
      return {
          'effective_gate_mean': stats['effective_gate_sum'] / count,
          'effective_gate_min': stats['effective_gate_min'],
          'effective_gate_max': stats['effective_gate_max'],
          'task_signal_mean': stats['task_signal_sum'] / count,
      }

  def _record_task_gate_stats(self, gate, gate_signal):
      if self._task_gate_stats is None:
          return

      gate_value = gate.detach().float().mean().item()
      signal_value = gate_signal.detach().float().mean().item()
      stats = self._task_gate_stats
      stats['effective_gate_sum'] += gate_value
      stats['task_signal_sum'] += signal_value
      stats['count'] += 1

      if stats['effective_gate_min'] is None:
          stats['effective_gate_min'] = gate_value
          stats['effective_gate_max'] = gate_value
      else:
          stats['effective_gate_min'] = min(stats['effective_gate_min'], gate_value)
          stats['effective_gate_max'] = max(stats['effective_gate_max'], gate_value)

  def _inner_forward(self, x, params, episode, return_feat=False):
    """ Forward pass for the inner loop. """
    feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    if return_feat:
      return logits, feat
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
          use_gradient_transport=False,
          task_gate_args=None):
    """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    task_gate_args = task_gate_args or {}
    with torch.enable_grad():
      # forward pass
      # AGAG Loss'ların bulunduğu yer.
      use_proto_signal = self._uses_prototype_task_signal(task_gate_args)
      if use_proto_signal:
        logits, support_feat = self._inner_forward(
          x, params, episode, return_feat=True)
        proto_gate_signal = self._compute_prototype_gate_signal(
          support_feat, y, task_gate_args)
      else:
        logits = self._inner_forward(x, params, episode)
        proto_gate_signal = None
      loss = F.cross_entropy(logits, y)
      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          gate_signal_grad = grad
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
              gate_logit = self.gradient_transport_logits[gate_key]
              if task_gate_args.get('enabled', False):
                  gate_signal = self._compute_task_gate_signal(
                      gate_signal_grad,
                      gate_logit,
                      task_gate_args,
                      proto_gate_signal)
                  residual_scale = task_gate_args.get('residual_scale', 1.0)
                  gamma_scale = task_gate_args.get('gamma_scale', 1.0)
                  signal_scale = task_gate_args.get('signal_scale', 1.0)
                  gate_logit = gate_logit + \
                      residual_scale * gamma_scale * \
                      self.task_gate_gammas[gate_key] * \
                      signal_scale * gate_signal
              gate = torch.sigmoid(gate_logit)
              gate_min = task_gate_args.get('gate_min', None) \
                  if task_gate_args.get('enabled', False) else None
              if gate_min is not None:
                  gate = gate.clamp_min(float(gate_min))
              if task_gate_args.get('enabled', False) and not detach:
                  self._record_task_gate_stats(gate, gate_signal)
              transported_grad = gate * grad
              updated_param = param - lr * transported_grad
          else:
              updated_param = param - lr * grad  #AGAG θi′=θi−α∇θiLsupport(θ)
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
          use_gradient_transport=False,
          task_gate_args=None):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)  #AGAG ilgili epizotta örnek sayısıyla etiket sayısı eşit mi kontrolü

    # Initializes a dictionary of momentum buffer for gradient descent in the 
    # inner loop. It has the same set of keys as the parameter dictionary.
    task_gate_args = task_gate_args or {}

    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:  #AGAG Klasik gradient descent yerine momentum gradient descent. Normal MAML'da yok, ufak bir ekleme. Çok önemli değil
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    #AGAG aşağıdaki self.efficient true ise daha az ram kullanmak amacıyla checkpoint mantığının çalıştırılması için kullanılan bir şey. Normalde kullanılmıyor.
    def _inner_iter_cp(episode, *state):
      """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach,
        use_gradient_transport=use_gradient_transport,
        task_gate_args=task_gate_args)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']): #AGAG buradaki step, bir taskteki support set üzerindeki verileri kaç kere işleyip kaç kere gradient'i güncelleyeceğimizi belirler.
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._inner_iter( #AGAG task için tek bir iterasyon
          x, y, params, mom_buffer, episode, inner_args, not meta_train,
          use_gradient_transport=use_gradient_transport,
          task_gate_args=task_gate_args)
        
    return params

  def forward(
          self,
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train,
          y_query=None,
          return_metrics=False,
          use_alignment_pre_loss=False,  # pre-alignment loss aktif mi?
          use_alignment_post_loss=False,  # post-alignment loss aktif mi?
          alignment_pre_weight=0.0,  # pre-alignment loss katsayısı (eta)
          alignment_post_weight=0.0,  # post-alignment loss katsayısı (eta)
          use_gradient_transport=False,
          task_gate_args=None
  ):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)
    task_gate_args = task_gate_args or {}
    if use_gradient_transport and task_gate_args.get('enabled', False) and \
      task_gate_args.get('collect_stats', True):
      self.reset_task_gate_stats()
    else:
      self._task_gate_stats = None

    # Alignment metrik/log hesaplaması yapılacak mı?
    # Bunun için hem return_metrics açık olmalı hem de query etiketleri gelmiş olmalı.
    do_alignment_log = return_metrics and (y_query is not None)

    # Pre-alignment loss gerçekten train loss'una eklenecek mi?
    # Sadece meta-train sırasında anlamlı.
    do_alignment_pre_loss = use_alignment_pre_loss and meta_train and (y_query is not None)

    # Post-alignment loss gerçekten train loss'una eklenecek mi?
    # Sadece meta-train sırasında anlamlı.
    do_alignment_post_loss = use_alignment_post_loss and meta_train and (y_query is not None)

    # Epoch boyunca episode bazlı pre-alignment loss değerlerini tutacağız.
    align_pre_loss_list = []

    # Epoch boyunca episode bazlı post-alignment loss değerlerini tutacağız.
    align_post_loss_list = []

    align_pre_list = []
    align_post_list = []
    # a dictionary of parameters that will be updated in the inner loop
    #AGAG Gereksiz yani gradyanı hesaplanmayacak parametrelerin çıkartılması
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
        any(s in name for s in inner_args['frozen'] + [
          'temp', 'gradient_transport_logits', 'task_gate_gammas']):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)): #AGAG x_shot.size(0) -> n_episode yani o batch içerisinde kaç tane task olduğu.
      # inner-loop training
      ##AGAG train moduna alınması dropout, batch normalization gibi şeylerin.
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()

      g_sup_pre = None
      # PRE alignment bloğuna gerçekten ihtiyaç var mı?
      # - log alacaksak lazım
      # - pre-loss kullanacaksak lazım
      # - post-loss kullanacaksak support gradient'ini daha sonra da kullanacağız
      need_pre_block = do_alignment_log or do_alignment_pre_loss or do_alignment_post_loss
      if need_pre_block:
          with torch.enable_grad():
              # Support ve query loss'larını başlangıç parametresi (theta) üzerinde hesaplıyoruz.
              logits_sup_pre = self._inner_forward(x_shot[ep], params, ep)
              loss_sup_pre = F.cross_entropy(logits_sup_pre, y_shot[ep])

              logits_qry_pre = self._inner_forward(x_query[ep], params, ep)
              loss_qry_pre = F.cross_entropy(logits_qry_pre, y_query[ep])

              param_list = list(params.values())

              # Support gradient'i:
              # Eğer pre-loss veya post-loss kullanacaksak graph korunmalı.
              keep_sup_grad_graph = do_alignment_pre_loss or do_alignment_post_loss
              g_sup_pre = self._get_grads_from_loss(
                  loss_sup_pre,
                  param_list,
                  retain_graph=keep_sup_grad_graph,
                  create_graph=keep_sup_grad_graph,
                  detach_grads=(not keep_sup_grad_graph)
              )

              # Query-pre gradient'i:
              # Sadece pre-loss aktifse graph'lı tutmamız gerekir.
              g_qry_pre = self._get_grads_from_loss(
                  loss_qry_pre,
                  param_list,
                  retain_graph=do_alignment_pre_loss,
                  create_graph=do_alignment_pre_loss,
                  detach_grads=(not do_alignment_pre_loss)
              )

              # PRE alignment cosine değeri
              align_pre = self._cosine_between_grad_lists(g_sup_pre, g_qry_pre)

              # Sadece log açıksa metriği kaydet
              if do_alignment_log:
                  align_pre_list.append(align_pre.detach().item())

              # Sadece pre-loss açıksa ceza terimini oluştur
              # Omega_align_pre = eta * (1 - cos)
              if do_alignment_pre_loss:
                  align_pre_loss = alignment_pre_weight * (1.0 - align_pre)
                  align_pre_loss_list.append(align_pre_loss)
      updated_params = self._adapt(  #AGAG Modelin inner loop'u -> θ′=θ−α∇θLsupport
        x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train,
        use_gradient_transport=use_gradient_transport,
        task_gate_args=task_gate_args)
      # inner-loop validation
      # Query tarafında gradient gerekip gerekmediğini belirliyoruz.
      # - log alacaksak lazım
      # - post-loss kullanacaksak da lazım
      need_post_block = do_alignment_log or do_alignment_post_loss

      # Eğer post alignment hesaplanacaksa burada mutlaka grad açık olmalı.
      # Aksi halde query gradient'ini çıkaramayız.
      grad_ctx = torch.enable_grad() if need_post_block else torch.set_grad_enabled(meta_train)
      # inner-loop validation / query evaluation
      with grad_ctx:
          self.eval()
          logits_ep = self._inner_forward(x_query[ep], updated_params, ep)

          # POST alignment bloğuna ihtiyaç var mı?
          if need_post_block:
              # Query loss'unu support update sonrası parametreler (theta') üzerinde hesaplıyoruz.
              loss_qry_post = F.cross_entropy(logits_ep, y_query[ep])

              # Post-loss aktifse gradient graph'ını korumamız gerekir.
              g_qry_post = self._get_grads_from_loss(
                  loss_qry_post,
                  list(updated_params.values()),
                  retain_graph=meta_train,
                  create_graph=do_alignment_post_loss,
                  detach_grads=(not do_alignment_post_loss)
              )

              # POST alignment cosine değeri:
              # support'taki başlangıç gradient'i ile update sonrası query gradient'ini karşılaştırıyoruz.
              align_post = self._cosine_between_grad_lists(g_sup_pre, g_qry_post)

              # Sadece log açıksa metriği kaydet
              if do_alignment_log:
                  align_post_list.append(align_post.detach().item())

              # Sadece post-loss açıksa ceza terimini oluştur
              # Omega_align_post = eta * (1 - cos)
              if do_alignment_post_loss:
                  align_post_loss = alignment_post_weight * (1.0 - align_post)
                  align_post_loss_list.append(align_post_loss)
      logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)
    if return_metrics:
        metrics = {
            'align_pre_mean': sum(align_pre_list) / len(align_pre_list) if len(align_pre_list) > 0 else None,
            'align_post_mean': sum(align_post_list) / len(align_post_list) if len(align_post_list) > 0 else None,

            # Şimdilik bilgi amaçlı döndürüyoruz.
            # Train.py tarafında ister loglarız ister loss'a ekleriz.
            'align_pre_loss_mean': torch.stack(align_pre_loss_list).mean() if len(align_pre_loss_list) > 0 else None,
            'align_post_loss_mean': torch.stack(align_post_loss_list).mean() if len(align_post_loss_list) > 0 else None,
        }
        return logits, metrics
    return logits

  #AGAG ALIGNMENT LOGLARI İÇİN EKLENEN FONKSİYONLAR
  def _uses_prototype_task_signal(self, task_gate_args):
      return (
          task_gate_args.get('enabled', False) and
          task_gate_args.get('signal') == 'proto_separability')

  def _compute_prototype_gate_signal(self, feat, y, task_gate_args):
      feat_for_signal = feat
      if task_gate_args.get('detach_signal', True):
          feat_for_signal = feat_for_signal.detach()
      feat_for_signal = feat_for_signal.float()

      if feat_for_signal.dim() > 2:
          feat_for_signal = feat_for_signal.flatten(1)

      if task_gate_args.get('proto_normalize_features', True):
          feat_for_signal = F.normalize(feat_for_signal, p=2, dim=1)

      classes = torch.unique(y)
      prototypes = []
      intra_distances = []
      for cls in classes:
          cls_feat = feat_for_signal[y == cls]
          proto = cls_feat.mean(dim=0)
          prototypes.append(proto)
          if cls_feat.size(0) > 1:
              cls_dist = (cls_feat - proto).pow(2).sum(dim=1).sqrt()
              intra_distances.append(cls_dist.mean())
          else:
              intra_distances.append(cls_feat.new_tensor(0.0))

      prototypes = torch.stack(prototypes)
      if prototypes.size(0) > 1:
          d_inter = torch.pdist(prototypes, p=2).mean()
      else:
          d_inter = prototypes.new_tensor(0.0)
      d_intra = torch.stack(intra_distances).mean()

      eps = float(task_gate_args.get('proto_eps', 1e-6))
      temp = float(task_gate_args.get('proto_temperature', 1.0))
      metric = task_gate_args.get('proto_metric', 'log_ratio')
      eps_t = feat_for_signal.new_tensor(eps)

      if metric == 'log_ratio':
          intra_floor = float(task_gate_args.get('proto_intra_floor', 0.5))
          denom = d_intra + feat_for_signal.new_tensor(intra_floor)
          raw_signal = torch.log((d_inter + eps_t) / (denom + eps_t))
      elif metric == 'margin':
          raw_signal = d_inter - d_intra
      elif metric == 'ratio':
          intra_floor = float(task_gate_args.get('proto_intra_floor', 0.5))
          raw_signal = d_inter / (
              d_intra + feat_for_signal.new_tensor(intra_floor) + eps_t)
      else:
          raise ValueError('invalid prototype gate metric: {}'.format(metric))

      return torch.tanh(raw_signal / max(temp, eps))

  def _compute_task_gate_signal(
          self,
          grad,
          gate_logit,
          task_gate_args,
          proto_gate_signal=None):
      signal = task_gate_args.get('signal', 'grad_norm')
      if signal == 'proto_separability':
          if proto_gate_signal is None:
              raise ValueError('prototype gate signal was not computed')
          value = proto_gate_signal
          if task_gate_args.get('detach_signal', True):
              value = value.detach()
          return value.to(device=gate_logit.device, dtype=gate_logit.dtype)

      if signal != 'grad_norm':
          raise ValueError('invalid task gate signal: {}'.format(signal))

      grad_for_signal = grad
      if task_gate_args.get('detach_signal', True):
          grad_for_signal = grad.detach()

      norm = torch.norm(grad_for_signal)
      if task_gate_args.get('normalize_by_numel', True):
          norm = norm / (grad_for_signal.numel() ** 0.5)

      value = torch.log1p(norm)
      return value.to(device=gate_logit.device, dtype=gate_logit.dtype)

  def _get_grads_from_loss(
          self,
          loss,
          params,
          retain_graph=False,
          create_graph=False,
          detach_grads=True
  ):
      """
      Bir loss'tan gradient listesi çıkarır.

      Args:
        loss: türev alınacak loss değeri.
        params: gradient'i alınacak parametre listesi.
        retain_graph: mevcut graph sonradan tekrar kullanılacak mı?
        create_graph: gradient'in de türevini alabilmek için graph oluşturulsun mu?
        detach_grads: sadece log/metric için kullanıyorsak gradient'i graph'tan kopar.
                      Loss olarak kullanacaksak False olmalı.
      """
      grads = autograd.grad(
          loss,
          params,
          retain_graph=retain_graph,
          create_graph=create_graph,
          allow_unused=True
      )

      out = []
      for p, g in zip(params, grads):
          # Bazı parametreler için grad None gelebilir.
          # Bu durumda aynı boyutta sıfır tensor koyuyoruz.
          if g is None:
              z = torch.zeros_like(p)
              out.append(z.detach() if detach_grads else z)
          else:
              out.append(g.detach() if detach_grads else g)
      return out

  def _flatten_grads(self, grads):
      return torch.cat([g.reshape(-1) for g in grads])

  def _cosine_between_grad_lists(self, grads1, grads2, eps=1e-12):
      v1 = self._flatten_grads(grads1)
      v2 = self._flatten_grads(grads2)
      return F.cosine_similarity(v1, v2, dim=0, eps=eps)
