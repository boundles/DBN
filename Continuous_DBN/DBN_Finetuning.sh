#!/bin/bash

THEANO_FLAGS='floatX=float32,device=gpu,nvcc.fastmath=True'  python DBN_Finetuning.py
