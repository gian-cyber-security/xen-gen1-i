from __future__ import annotations
import argparse,math
from pathlib import Path
import imageio.v3 as iio,numpy as np,torch
from PIL import Image
from model.tokenizer import XENTokenizer
from model.image_model import XENImageModel
from model.image_conditioner import XENImageTextEncoder
from tools.agent_runtime import build_runtime_context, remember_turn
def ab(t): return torch.cos(((t.float()/999)+.008)/1.008*math.pi/2).pow(2).clamp(1e-4,.9999)
def load_image(path,size,device):
 im=Image.open(path).convert("RGB").resize((size,size)); arr=np.asarray(im)
 return torch.from_numpy(arr).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).to(device)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description="XEN-GEN1-I text-to-image and image editing")
 p.add_argument("--model-dir",default="outputs/xen-gen1-i"); p.add_argument("--prompt",required=True); p.add_argument("--image")
 p.add_argument("--strength",type=float,default=.65); p.add_argument("--output",default="outputs/xen-gen1-i/generated.png")
 p.add_argument("--size",type=int,default=256); p.add_argument("--steps",type=int,default=50); p.add_argument("--seed",type=int,default=42)
 p.add_argument("--web-search",action="store_true"); p.add_argument("--no-web-search",action="store_true")
 p.add_argument("--memory-file",default="outputs/xen_memory.json"); p.add_argument("--no-memory",action="store_true"); a=p.parse_args()
 prompt=a.prompt.strip()
 runtime,_=build_runtime_context(prompt,None if a.no_memory else a.memory_file,force_web=a.web_search,no_web=a.no_web_search)
 if runtime: prompt=runtime+"\n\nUSER PROMPT:\n"+prompt
 if not .05<=a.strength<=1: raise ValueError("--strength must be between 0.05 and 1.0")
 d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); z=torch.load(Path(a.model_dir)/"model.pt",map_location=d,weights_only=False)
 t=XENTokenizer.load(Path(a.model_dir)/"tokenizer.json"); m=XENImageModel().to(d); c=XENImageTextEncoder().to(d)
 m.load_state_dict(z["model"]); c.load_state_dict(z["conditioner"]); m.eval(); c.eval()
 ids=torch.tensor([t.encode(prompt,max_length=256)],device=d); cond=c(ids); g=torch.Generator(device=d).manual_seed(a.seed)
 if a.image:
  base=load_image(a.image,a.size,d); start=max(1,min(999,round(999*a.strength))); ts=torch.linspace(start,0,a.steps,device=d)
  x=ab(torch.tensor([start],device=d)).sqrt()[:,None,None,None]*base+(1-ab(torch.tensor([start],device=d))).sqrt()[:,None,None,None]*torch.randn(base.shape,device=d,generator=g)
 else: ts=torch.linspace(999,0,a.steps,device=d); x=torch.randn((1,3,a.size,a.size),device=d,generator=g)
 for i in range(len(ts)-1):
  q=ts[i].expand(1); qn=ts[i+1].expand(1); aq=ab(q)[:,None,None,None]; an=ab(qn)[:,None,None,None]; e=m(x,q,cond); x0=((x-(1-aq).sqrt()*e)/aq.sqrt().clamp_min(1e-4)).clamp(-1.5,1.5); x=an.sqrt()*x0+(1-an).sqrt()*e
 arr=((x[0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); iio.imwrite(a.output,arr)
 print("Saved image:",a.output)
 if not a.no_memory: remember_turn(a.prompt,"Generated image: "+a.output,a.memory_file)
if __name__=="__main__": main()
