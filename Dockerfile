FROM python:3.9.7

ADD helpers.py .
ADD simple_inference.py .

RUN pip install ftfy
RUN pip install regex
RUN pip install tqdm
RUN pip install torch
RUN pip install torchvision
RUN pip install transformers
RUN pip install git+https://github.com/openai/CLIP.git

COPY ["M-BERT-Base-69-ViT Linear Weights.pkl", "/"]