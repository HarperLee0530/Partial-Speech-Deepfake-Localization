"""Main coarse supervision only. Architecture and inference are unchanged."""
import torch
from torch.nn import functional as F
from cnn_ps.train_dual import training_loss as original_loss

def aggregate(logits, durations, mode):
    if logits.shape != durations.shape or logits.ndim != 2 or logits.shape[1] % 8:
        raise ValueError("Aligned B,T grids divisible by eight required")
    z=logits.reshape(logits.shape[0],-1,8)
    d=durations.reshape_as(z)
    if mode=="A0":
        return z.mean(-1)
    valid=d>0
    safe=torch.where(valid,z,torch.zeros_like(z))
    total=d.sum(-1,keepdim=True)
    w=d/total.clamp_min(1e-12)
    if mode=="M0":
        return (safe*w).sum(-1)
    if mode!="R1":
        raise ValueError(mode)
    # Give empty cells one dummy term, then discard them from both loss and gradients.
    occupied=total>0
    dummy=torch.zeros_like(valid);dummy[...,0]=~occupied.squeeze(-1)
    logw=torch.where(valid,w.clamp_min(1e-30).log(),torch.full_like(w,-torch.inf))
    logw=torch.where(dummy,torch.zeros_like(logw),logw)
    value=torch.logsumexp(safe+logw,dim=-1) # fixed tau=1.0
    return torch.where(occupied.squeeze(-1),value,torch.zeros_like(value))

def coarse_loss(final,batch,mode):
    durations=batch["fine_weights"].to(final.device)
    pooled=aggregate(final,durations,mode)
    w=batch["weights"].to(final.device)
    y=batch["labels"].to(final.device)
    valid=w>0
    return (F.binary_cross_entropy_with_logits(pooled[valid],y[valid],reduction="none")*w[valid]).sum()/w[valid].sum()

def training_loss(final,outputs,batch,device,boundary_enabled):
    mode=ACTIVE_ARM
    total,parts=original_loss(final,outputs,batch,device,boundary_enabled)
    old=coarse_loss(final,batch,"A0")
    if mode=="A0": return total,parts
    new=final.new_zeros(()) if mode=="S1" else coarse_loss(final,batch,mode)
    total=total-old+new
    parts.update(main=parts["main"]-float(old.detach())+float(new.detach()),
                 main160=float(new.detach()), historical_main160=float(old.detach()))
    return total,parts

ACTIVE_ARM="A0"
