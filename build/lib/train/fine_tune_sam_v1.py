import os, cv2, json, argparse, logging, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
from mtcnn import MTCNN
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class CelebAMaskHQDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prompt_dir):
        self.image_dir, self.mask_dir, self.prompt_dir = image_dir, mask_dir, prompt_dir
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        name = self.image_files[idx]
        img_bgr = cv2.imread(os.path.join(self.image_dir, name))
        if img_bgr is None: return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_path = os.path.join(self.mask_dir, name.replace(".jpg", ".png"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST) if mask is not None else np.zeros((1024,1024), np.uint8)
        prompt_path = os.path.join(self.prompt_dir, name.replace(".jpg", ".json"))
        if not os.path.exists(prompt_path): return None
        prompt = json.load(open(prompt_path))[0]
        return (torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255,
                torch.tensor(mask, dtype=torch.float32)/255,
                torch.tensor(prompt, dtype=torch.float32))

def dice_loss(pred, target, eps=1e-6):
    p = torch.sigmoid(pred)
    inter = (p*target).sum((1,2,3))
    union = p.sum((1,2,3))+target.sum((1,2,3))
    return 1-(2*inter+eps)/(union+eps)

def clean_mask(m):
    m = cv2.GaussianBlur(m.astype(np.float32),(5,5),0)
    _,m=cv2.threshold(m,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8,cv2.CV_32S)
    if n>1:
        largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
        m=(labels==largest).astype(np.uint8)
    return m

def fine_tune_sam(ckpt,img_dir,mask_dir,prompt_dir,epochs,batch_size,patience):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam=sam_model_registry["vit_b"](checkpoint=ckpt).to(device)
    for p in sam.image_encoder.parameters(): p.requires_grad=False
    sam.train()
    ds=CelebAMaskHQDataset(img_dir,mask_dir,prompt_dir)
    def collate(batch):
        batch=[b for b in batch if b is not None]
        if not batch: return None
        return tuple(torch.stack(t) for t in zip(*batch))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    opt=torch.optim.Adam(list(sam.prompt_encoder.parameters())+list(sam.mask_decoder.parameters()),lr=1e-5)
    scaler=GradScaler(); best=float("inf"); wait=0
    for ep in range(epochs):
        tot=0
        for batch in tqdm(dl,leave=False):
            if batch is None: continue
            imgs,masks,boxes=[t.to(device) for t in batch]
            loss_sum=0
            for i in range(imgs.size(0)):
                with autocast():
                    emb=sam.image_encoder(imgs[i:i+1])
                    sp,de=sam.prompt_encoder(points=None,boxes=boxes[i][None,None,:],masks=None)
                    low,_=sam.mask_decoder(image_embeddings=emb,image_pe=sam.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=sp,dense_prompt_embeddings=de,
                                           multimask_output=False)
                    up=torch.nn.functional.interpolate(low,(1024,1024),mode="nearest")
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(up.squeeze(1),masks[i:i+1])
                    dcl=dice_loss(up,masks[i:i+1])
                    loss=0.5*bce+0.5*dcl
                    loss_sum+=loss
            opt.zero_grad(); scaler.scale(loss_sum).backward(); scaler.step(opt); scaler.update()
            tot+=loss_sum.item()
        avg=tot/len(dl); logging.info(f"epoch {ep+1} loss {avg:.4f}")
        if avg<best:
            best=avg; wait=0
            torch.save(sam.state_dict(),"checkpoints/best_fine_tuned_sam.pth")
        else:
            wait+=1
            if wait>=patience: break
    torch.save(sam.state_dict(),"checkpoints/fine_tuned_sam_last.pth")

def segment_face(image_path,ckpt):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam=sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.load_state_dict(torch.load("checkpoints/best_fine_tuned_sam.pth",map_location=device))
    sam.eval().to(device)
    pred=SamPredictor(sam); mt=MTCNN()
    bgr=cv2.imread(image_path); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    faces=mt.detect_faces(rgb)
    if not faces: return
    x,y,w,h=faces[0]["box"]; pad=.3
    x=max(0,int(x-w*pad)); y=max(0,int(y-h*pad))
    w=min(rgb.shape[1]-x,int(w*(1+2*pad))); h=min(rgb.shape[0]-y,int(h*(1+2*pad)))
    box=np.array([x,y,x+w,y+h])
    pred.set_image(rgb)
    masks,_,_=pred.predict(box=box,multimask_output=False)
    mask=clean_mask(masks[0].astype(np.uint8))
    out=bgr.copy(); out[mask==0]=255
    cv2.imwrite("output.jpg",out)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["train","infer"],required=True)
    ap.add_argument("--image_dir", default="data/train/images")
    ap.add_argument("--mask_dir", default="data/train/masks")
    ap.add_argument("--prompt_dir", default="data/train/prompts")
    ap.add_argument("--model_checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--image_path")
    ap.add_argument("--batch_size",type=int,default=16)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--patience",type=int,default=2)
    args=ap.parse_args()
    os.makedirs("checkpoints",exist_ok=True)
    if args.mode=="train":
        fine_tune_sam(args.model_checkpoint,args.image_dir,args.mask_dir,args.prompt_dir,
                      args.epochs,args.batch_size,args.patience)
    else:
        segment_face(args.image_path,args.model_checkpoint)
