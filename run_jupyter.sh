#!/bin/bash
source activate capstone
CUDA_VISIBLE_DEVICES=0 jupyter notebook --no-browser --port 9999
