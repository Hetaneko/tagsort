import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
import requests
import os
import networks
import gradio as gr
import subprocess
import modules
import base64

from llama_cpp import Llama

from modules.api.models import *
from modules.api import api

text2 = """<｜User｜>

"""
text1 = """

go through every tag in this prompt and categorize every tag in this form:
Clothing (any clothing/wearable accessory/wearable tag)
Expression (any physical reaction or expression, dont count actions that is not related to clothing)
Characteristic (any hair/eye/body/character/irl reference related tags, dont count simple terms like "ass" or "thighs" without descriptors. it needs to be like "huge ass" or "thick thighs")

if the tag fits one of these categories say the name in capital letters if it fits to none of these categories say "NONE"

<｜Assistant｜>
<think>
Okay, so I've got this prompt with a bunch of tags, and I need to go through each one and categorize them into Clothing, Expression, or Characteristic. If none of those fit, I just say "NONE". Let me take it step by step.

Alright, lets go through each one.

"""
os.environ["CUDA_VISIBLE_DEVICES"]="1"
llm = Llama(
      model_path="/tmp/lmmodel/DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf",
      n_gpu_layers=-1,
      n_ctx=8192,
      usecpu =False,
      usecublas=['normal', 'mmq'],
      main_gpu=1)

def sorttags_api(_: gr.Blocks, app: FastAPI):
    @app.post("/mikww/sorttags")
    async def sorttags(
        tags: str = Body("none", title='tags')
    ):
        splittagz = tags.split(",")
        promptt = text2 + tags + text1
        for idx,x in enumerate(splittagz):
          promptt += str(idx + 1) + ". **"+x+"**:"
          if idx == len(splittagz) - 1:
            outputz = llm(promptt,
              max_tokens=None,
              stop=["</think>"],
              # echo=True,
              temperature=0.75,
              top_p=0.92,
              top_k=100,
              seed=1
            )
            promptt = promptt + outputz["choices"][0]["text"] + "</think>"
          else:
            outputz = llm(promptt,
              max_tokens=None,
              stop=["\n\n"],
              # echo=True,
              temperature=0.75,
              top_p=0.92,
              top_k=100,
              seed=1
            )
            promptt = promptt + outputz["choices"][0]["text"] + "\n\n"
        tagz = []
        promptt = promptt + "\n\nHere is the categorization of each tag from the prompt:\n\n"
        for idx,x in enumerate(splittagz):
          promptt += str(idx + 1) + ". **"+x+"** - "
          outputz = llm(promptt,
            max_tokens=None,
            stop=["\n"],
            # echo=True,
            temperature=0.75,
            top_p=0.92,
            top_k=100,
            seed=1
          )
          rez = outputz["choices"][0]["text"]
          tagz.append({"tag":x,"category":rez})
          promptt = promptt + outputz["choices"][0]["text"] + "\n"
        return {"Tags":tagz}
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(sorttags_api)
except:
    pass
