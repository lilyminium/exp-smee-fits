#!/bin/bash

conda activate smee-stack-dev

conda env export > environment.yml

mkdir logs

python download-optimizations.py > logs/download-optimizations.txt 2>&1
