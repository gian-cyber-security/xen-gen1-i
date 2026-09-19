from __future__ import annotations
import math,torch
from torch import nn
import torch.nn.functional as F
class SinusoidalTimeEmbedding(nn.Module):
 def __init__(self,dim): super().__init__(); self.dim=dim
 def forward(self,t):
  half=self.dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1)); e=t.float()[:,None]*f[None,:]; return torch.cat([e.sin(),e.cos()],-1)
class ResBlock(nn.Module):
 def __init__(self,ic,oc,td,cd):
  super().__init__(); self.n1=nn.GroupNorm(8,ic); self.c1=nn.Conv2d(ic,oc,3,padding=1); self.n2=nn.GroupNorm(8,oc); self.c2=nn.Conv2d(oc,oc,3,padding=1); self.t=nn.Linear(td,oc); self.c=nn.Linear(cd,oc); self.s=nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity()
 def forward(self,x,t,c):
  h=self.c1(F.silu(self.n1(x))); h=h+self.t(t)[:,:,None,None]+self.c(c)[:,:,None,None]; return self.c2(F.silu(self.n2(h)))+self.s(x)
class XENImageModel(nn.Module):
 def __init__(self,cond_dim=256,base=64):
  super().__init__(); td=base*4; self.time=SinusoidalTimeEmbedding(td); self.text_proj=nn.Sequential(nn.Linear(cond_dim,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim)); self.i=nn.Conv2d(3,base,3,padding=1); self.d1=ResBlock(base,base,td,cond_dim); self.d2=ResBlock(base,base*2,td,cond_dim); self.d3=ResBlock(base*2,base*4,td,cond_dim); self.m1=ResBlock(base*4,base*4,td,cond_dim); self.m2=ResBlock(base*4,base*4,td,cond_dim); self.u3=ResBlock(base*8,base*2,td,cond_dim); self.u2=ResBlock(base*4,base,td,cond_dim); self.u1=ResBlock(base*2,base,td,cond_dim); self.o=nn.Conv2d(base,3,3,padding=1)
 def forward(self,x,timestep,text_embedding):
  c=self.text_proj(text_embedding); t=self.time(timestep); x0=self.i(x); d1=self.d1(x0,t,c); d2=self.d2(F.avg_pool2d(d1,2),t,c); d3=self.d3(F.avg_pool2d(d2,2),t,c); m=self.m2(self.m1(d3,t,c),t,c); u3=self.u3(torch.cat([F.interpolate(m,size=d2.shape[-2:],mode="nearest"),d2],1),t,c); u2=self.u2(torch.cat([F.interpolate(u3,size=d1.shape[-2:],mode="nearest"),d1],1),t,c); return self.o(F.silu(self.u1(torch.cat([u2,x0],1),t,c)))
