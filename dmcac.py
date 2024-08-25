
import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('gdrive')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import glob
from scipy.special import softmax
import numpy as np
import torch.nn.functional as F
import math
import os
from torch import optim
from tqdm import tqdm
import gc


base_model = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True)


def load1_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

augment1 = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
  transforms.RandomHorizontalFlip(p=0.5),
])

augment2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
])

augment3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_imageAugment(image_path):
    img = Image.open(image_path)
    img_tensor1 = augment1(img).unsqueeze(0)
    img_tensor2 = augment2(img).unsqueeze(0)
    img_tensor3 = augment3(img).unsqueeze(0)
    return img_tensor1,img_tensor2,img_tensor3

allImages = glob.glob('/content/gdrive/MyDrive/*.jpg')

def L2norm(vec):
  return vec/np.linalg.norm(vec,ord = 2)

Z_D = {}
Z_A = {}
labels = []

for i,a in enumerate(allImages):
  labels.append(i)
  image_tensor = load1_image(a)
  base_model.eval()
  with torch.no_grad():
    currentImageName = 'image' + str(i)
    normal_tensor = base_model(image_tensor)
    Z_D[currentImageName + '_normal'] = L2norm(normal_tensor)


"""
Full retrieval setting
"""
db_embeddings = np.array(list(Z_D.values()))
db_embeddings= db_embeddings.reshape(len(db_embeddings),1000)
db_embeddings = db_embeddings.T

query_embedding = np.array(list(Z_A.values()))
query_embedding = query_embedding.reshape(len(query_embedding),1000)

P = np.dot(query_embedding, db_embeddings)
row_sums = P.sum(axis=1, keepdims=True)
P = P / row_sums

P_prime = []
for p in P :
  P_prime.append(softmax(p))


"""
Approximate retrieval setting
"""
from sklearn.neighbors import NearestNeighbors


def approxRetrieve(Z_A,Z_D):
  retDict ={}
  allKeys = list(Z_D.keys())
  allValues = list(Z_D.values())
  allValues = np.array(allValues).reshape(len(allValues),1000)

  nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto')
  nbrs.fit(allValues)

  for k,e in zip(list(Z_A.keys()),list(Z_A.values())):
    e = e.reshape(1,-1)
    distances, indices = nbrs.kneighbors(e)
    retDict[k] = indices[0]
  return retDict

def getCombinations(Z_A,Z_D):
  retDict = approxRetrieve(Z_A,Z_D)
  allKeys = list(retDict.keys())
  allValues = list(retDict.values())

  S_union = {}
  S_unionEmbeddings = {}
  track = {}
  for i in allKeys:
    if i.split('_')[0] not in S_union: S_union[i.split('_')[0]] = retDict[i]
    else : S_union[i.split('_')[0]] = np.union1d(S_union[i.split('_')[0]],retDict[i])
    for r in retDict[i]:
        if i.split('_')[0] not in track : track[i.split('_')[0]] = []
        if r in track[i.split('_')[0]] : continue
        else :
          track[i.split('_')[0]].append(r)
          if i.split('_')[0] not in S_unionEmbeddings: S_unionEmbeddings[i.split('_')[0]] = [list(Z_D.values())[r]]
          else: S_unionEmbeddings[i.split('_')[0]].append(list(Z_D.values())[r])

  return S_union,S_unionEmbeddings


def getP_and_P_prime(Z_A,Z_D):
  S_union,S_unionEmbeddings = getCombinations(Z_A,Z_D) 
  currentKey = list(S_union.keys())[0]
  S_unionEmbeddings[currentKey] = np.array(S_unionEmbeddings[currentKey])
  S_unionEmbeddings[currentKey] = S_unionEmbeddings[currentKey].reshape(len(S_unionEmbeddings[currentKey]),1000)
  allValues = list(Z_A.values())
  allValues = np.array(allValues).reshape(len(allValues),1000)
  P = np.dot(allValues, S_unionEmbeddings[currentKey].T)
  row_sums = P.sum(axis=1, keepdims=True)
  P = P / row_sums
  P_prime = []
  for p in P :
    P_prime.append(softmax(torch.tensor(p).reshape(1,-1)))
  return P,P_prime

def KL_divergence(P,R):
  return np.sum(P * np.log(P / R))

def L(P,R):
  return ((KL_divergence(P,R) + KL_divergence(R,P))/2)

def L_frob(P_prime):
  A = len(P_prime)
  exterior = 0

  for i in range(2,A):
    interior = 0
    for j in range(1,i):
      interior += L(P_prime[i].cpu().detach().numpy(),P_prime[j].cpu().detach().numpy())
    exterior += interior
  return np.sqrt(exterior)


class dmcac_VIT(nn.Module):
    def __init__(self, num_labels=3):
        super(dmcac_VIT, self).__init__()
        self.model = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True)
        
        for param in self.model.parameters():
          param.requires_grad = True

    def forward(self,img_tensor):
    
      out = self.model(img_tensor)
      
      return out

model = dmcac_VIT(len(labels)).to('cuda')  


modelFolderpath = './models'
if not os.path.isdir(modelFolderpath):
  os.makedirs(modelFolderpath, exist_ok = True)


lr = 0.1
epoch = 200

all_network_params = list(model.parameters())
optimizer = optim.SGD(all_network_params, lr = lr,momentum=1.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)
criterion = nn.CrossEntropyLoss()


F = 1000 # len of output linear vector
Wq = nn.Parameter(torch.randn(F,F)).to('cuda')
Wk = nn.Parameter(torch.randn(F,F)).to('cuda')
Wv = nn.Parameter(torch.randn(F,F)).to('cuda')
Wc = nn.Parameter(torch.randn(F,len(labels))).to('cuda')
softmax = nn.Softmax(dim=1)

labels = torch.tensor(labels, dtype=torch.float).reshape(1,-1)



for e in tqdm(range(epoch)):
  model.train()

  for img in allImages:
    with torch.device('cuda'):
      with torch.set_grad_enabled(True):
          
          print(img)
          image_tensor = load1_image(img)
          img_tensor1,img_tensor2,img_tensor3 = load_imageAugment(img)

          
          normal_tensor = model(image_tensor.to('cuda'))
          out_tensor1,out_tensor2,out_tensor3 = model(img_tensor1.to('cuda')),model(img_tensor2.to('cuda')),model(img_tensor3.to('cuda'))
          
          Z_A = {}

          currentImageName = 'image0'
            
          Z_A[currentImageName + '_normal'] = L2norm(normal_tensor.cpu().detach().numpy())
          Z_A[currentImageName + '_a1'] = L2norm(out_tensor1.cpu().detach().numpy())
          Z_A[currentImageName + '_a2'] = L2norm(out_tensor2.cpu().detach().numpy())
          Z_A[currentImageName + '_a3'] = L2norm(out_tensor3.cpu().detach().numpy())

          S_union,S_unionEmbeddings = getCombinations(Z_A,Z_D)
          q_ce_all = []
          q_cac_all = [] 
          for z in list(Z_A.values()):
            z = torch.tensor(z)
            z = z.reshape(1000,1)
            q_ce_all.append((softmax(torch.matmul(Wc.T, z))).T)

          
            S_unionEmbeddings_tensors = []
            for s in list(list(S_unionEmbeddings.values())[0]):
              S_unionEmbeddings_tensors.append(torch.tensor(s))
            
            S_unionEmbeddings_tensors = torch.stack(S_unionEmbeddings_tensors).reshape(len(S_unionEmbeddings_tensors),1000)
            Q =  torch.matmul(Wq,z)
            K = torch.matmul(Wk,torch.tensor(S_unionEmbeddings_tensors).T)
            V = torch.matmul(Wv,torch.tensor(S_unionEmbeddings_tensors).T)
            z_prime = softmax((torch.matmul(Q.T,K))/math.sqrt(z.shape[0])*V).to('cuda')
            q_cac = softmax(torch.matmul(Wc.T,z_prime))
            q_cac = q_cac.sum(dim=0, keepdim=True)
            q_cac_all.append(q_cac)

            
          P,P_prime = getP_and_P_prime(Z_A,Z_D)
          Lfrob = L_frob(P_prime)

          L_CE = 0 
          L_CAC = 0 
          for ce,cac in zip(q_ce_all,q_cac_all):
            L_CE += criterion(ce,labels.to('cuda'))
            L_CAC += criterion(cac,labels.to('cuda'))
                    
          beta_frob,beta_ce,beta_cac = 1,1,1
          total_loss =  beta_frob*Lfrob + beta_ce*L_CE + beta_cac*L_CAC 
          print('total_loss',total_loss)
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          torch.cuda.empty_cache()
          gc.collect()
  exp_lr_scheduler.step()
  if e % 5 == 0  :
      model_save_name = f'model_{e}.pt'
      model_save_path = os.path.join(modelFolderpath, f'model_{e}.pt')
      torch.save(model.state_dict() , model_save_path)
