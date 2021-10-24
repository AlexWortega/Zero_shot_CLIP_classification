
import torch
from CLIP import clip
from PIL import Image
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
import pickle

import torch
import transformers


class MultilingualClip(torch.nn.Module):
    def __init__(self, model_name, tokenizer_name, head_name, weights_dir='/content/'):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.head_path = weights_dir + head_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        self.clip_head = torch.nn.Linear(in_features=768, out_features=640)
        self._load_head()

    def forward(self, txt):
        txt_tok = self.tokenizer(txt, padding=True, return_tensors='pt')
        
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)

    def _load_head(self):
        with open(self.head_path, 'rb') as f:
            lin_weights = pickle.loads(f.read())
        self.clip_head.weight = torch.nn.Parameter(torch.tensor(lin_weights[0]).float().t())
        self.clip_head.bias = torch.nn.Parameter(torch.tensor(lin_weights[1]).float())


AVAILABLE_MODELS = { 
    'M-BERT-Base-ViT-B': {
        'model_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'tokenizer_name': 'M-CLIP/M-BERT-Base-ViT-B',
        'head_name': 'M-BERT-Base-69-ViT Linear Weights.pkl'
    },
}


def load_model(name):
    config = AVAILABLE_MODELS[name]
    return MultilingualClip(**config)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
text_model = load_model('M-BERT-Base-ViT-B')




def max_cosine_text(image_path,labels):
  """
  image_path: путь к изображению для которго ищем
  text: текстовые метки для которы ищем
  """
  cos = nn.CosineSimilarity(dim=1, eps=1e-32)
  image= preprocess(Image.open(image_path)).unsqueeze(0).to(device)
  

  

  with torch.no_grad():
    max = 0
    label = ''

    for txt in labels:
      embed = text_model(txt)
      #print(embed.shape)
      

      image_features = model.encode_image(image)

      text_features = embed[:,:512]
      
      cosine=cos(image_features,text_features)
      if cosine>max:
        max = cosine
        label = txt
        

    
    return max,label



def max_cosine_images(image_paths,label):
  """
  image_paths: массив путей к файлам в которых ищем
  label : текстовое описание того что ищем 
  """
  cos = nn.CosineSimilarity(dim=1, eps=1e-32)
  
  

  embed = text_model(label)
  text_features = embed[:,:512]
  
  with torch.no_grad():
    max = 0
    max_score_image = None

    for image_p in image_paths:
      
      
      image= preprocess(Image.open(image_p)).unsqueeze(0).to(device)
  

      image_features = model.encode_image(image)

      
      
      cosine=cos(image_features,text_features)

      if cosine>max:
        max = cosine
        path = image_p
        max_score_image  = image
        

    
    return max, max_score_image,path 
    
