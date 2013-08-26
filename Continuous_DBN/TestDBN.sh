#!/bin/bash

THEANO_FLAGS='floatX=float32,device=gpu,nvcc.fastmath=True'  python TestDBN.py
