from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from model.image_model import XENImageModel
from model.image_conditioner import XENImageTextEncoder
from model.tokenizer import XENTokenizer
class DS(Dataset):
 def __init__(self,p,t,s):
  self.rows=[(x["image"],x["caption"]) for x in map(json.loads,open(p,encoding="utf-8")) if x]; self.t=t; self.s=s
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  p,c=self.rows[i]; im=Image.open(p).convert("RGB").resize((self.s,self.s)); x=torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/127.5-1; return x,torch.tensor(self.t.encode(c,max_length=256))
def collate(b):
 m=max(x[1].numel() for x in b); return torch.stack([x[0] for x in b]),torch.stack([F.pad(x[1],(0,m-x[1].numel())) for x in b])
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="datasets/image_data.jsonl"); p.add_argument("--output",default="outputs/xen-gen1-i"); p.add_argument("--size",type=int,default=256); p.add_argument("--batch-size",type=int,default=2); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--steps",type=int,default=10000); a=p.parse_args()
 t=XENTokenizer(); t.fit(); ds=DS(a.data,t,a.size); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True); d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); m=XENImageModel().to(d); c=XENImageTextEncoder().to(d); ps=list(m.parameters())+list(c.parameters()); o=torch.optim.AdamW(ps,lr=a.lr,weight_decay=.01); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); t.save(out/"tokenizer.json"); step=0
 while step<a.steps:
  for im,ids in dl:
   im,ids=im.to(d),ids.to(d); n=torch.randn_like(im); tt=torch.randint(0,1000,(im.shape[0],),device=d); ab=torch.cos(((tt.float()/999)+.008)/1.008*torch.pi/2).pow(2).clamp(1e-4,.9999)[:,None,None,None]; pred=m(ab.sqrt()*im+(1-ab).sqrt()*n,tt,c(ids)); loss=F.mse_loss(pred,n); o.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ps,1); o.step(); step+=1
   if step%20==0: print(f"step={step} loss={loss.item():.5f}")
   if step>=a.steps: break
 torch.save({"model":m.state_dict(),"conditioner":c.state_dict(),"size":a.size},""+str(out/"model.pt"))
if __name__=="__main__": main()
