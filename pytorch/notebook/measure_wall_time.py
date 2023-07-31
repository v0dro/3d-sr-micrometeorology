#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sys
from logging import INFO, StreamHandler, getLogger

logger = getLogger()
if not any(["StreamHandler" in str(handler) for handler in logger.handlers]):
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)


# # Import libraries

# In[3]:


import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from src.loss_maker import make_loss
from src.model_maker import make_model
from src.utils import set_seeds
from tqdm.notebook import tqdm

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)


# In[4]:


os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=False)


# # Define constants

# In[5]:


ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


# In[6]:


path = f"{ROOT_DIR}/pytorch/config/default.yml"
with open(path) as file:
    CONFIG = yaml.safe_load(file)


# In[7]:


DEVICE = "cuda"


# In[8]:


# tensor size for testing
UPSCALE_FACTOR = 4
BATCH_SIZE_FOR_TEST = 1
BATCH_SIZE_FOR_TRAIN = 1
NUM_CHANNELS = 4
NUM_Z = 32
NUM_Y_FOR_TEST = 320
NUM_X_FOR_TEST = 320
NUM_Y_FOR_TRAIN = 320
NUM_X_FOR_TRAIN = 320


# # Measure wall time to test

# In[9]:


model = make_model(CONFIG).to(DEVICE)
_ = model.eval()


# In[10]:


num_batches = 100  # arbitrary number
wall_times = []

for _ in tqdm(range(num_batches)):
    Xs = torch.randn(
        (
            BATCH_SIZE_FOR_TEST,
            NUM_CHANNELS,
            NUM_Z // UPSCALE_FACTOR,
            NUM_Y_FOR_TEST // UPSCALE_FACTOR,
            NUM_X_FOR_TEST // UPSCALE_FACTOR,
        ),
        device=DEVICE,
        dtype=torch.float32,
    )
    bs = torch.randn(
        (BATCH_SIZE_FOR_TEST, 1, NUM_Z, NUM_Y_FOR_TEST, NUM_X_FOR_TEST),
        device=DEVICE,
        dtype=torch.float32,
    )
    # building data has one channel

    start = time.time()
    with torch.no_grad():
        preds = model(Xs, bs)
    end = time.time()

    wall_times.append(end - start)


# In[11]:


print("Train: ", str(np.sum(wall_times)))  # seconds


# # Measure wall time to train

# In[12]:


model = make_model(CONFIG).to(DEVICE)
_ = model.train()


# In[13]:


loss_fn = make_loss(CONFIG)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["train"]["lr"])


# In[14]:


num_batches = 100  # arbitrary number
wall_times = []

for _ in tqdm(range(num_batches)):
    ys = torch.randn(
        (
            BATCH_SIZE_FOR_TRAIN,
            NUM_CHANNELS,
            NUM_Z,
            NUM_Y_FOR_TRAIN,
            NUM_X_FOR_TRAIN,
        ),
        device=DEVICE,
        dtype=torch.float32,
    )
    Xs = torch.randn(
        (
            BATCH_SIZE_FOR_TRAIN,
            NUM_CHANNELS,
            NUM_Z // UPSCALE_FACTOR,
            NUM_Y_FOR_TRAIN // UPSCALE_FACTOR,
            NUM_X_FOR_TRAIN // UPSCALE_FACTOR,
        ),
        device=DEVICE,
        dtype=torch.float32,
    )
    bs = torch.randn(
        (BATCH_SIZE_FOR_TRAIN, 1, NUM_Z, NUM_Y_FOR_TRAIN, NUM_X_FOR_TRAIN),
        device=DEVICE,
        dtype=torch.float32,
    )
    # building data has one channel

    start = time.time()

    preds = model(Xs, bs)
    loss = loss_fn(preds, ys, bs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end = time.time()

    wall_times.append(end - start)


# In[15]:


print("Infer: ", str(np.sum(wall_times)))  # seconds

