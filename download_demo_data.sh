#!/bin/bash

# Define a download function
function google_drive_download()
{
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Prepare directories
mkdir -p ./images ./meshes ./demo/guitar/encoder_dino ./demo/guitar/encoder_sam ./demo/guitar/bsb

# Image
google_drive_download 1WnWqus0QmAEPO9Ftsja2hSfN963fskmP guitar_01.png
mv guitar_01.png ./images

# Mesh
google_drive_download 1USul1CkApiCEDYbXBnRslhKna_BsQCfw guitar.obj
mv guitar.obj ./meshes

# Per-vertex DINOv2 encoder features
google_drive_download 1YeZy5rQpsqjsw2ws4R4rdDVjfK9FNyBJ pred_f.pth
mv pred_f.pth ./demo/guitar/encoder_dino

# Per-vertex SAM encoder features
google_drive_download 108-tioyNq1NwZgiPCN2OfPIbpuV8Y6gf pred_f.pth
mv pred_f.pth ./demo/guitar/encoder_sam

# iSeg Decoder checkpoint
google_drive_download 16qk2At1dNUsoC0lMfFovsfy1SUgJT2Ia decoder_checkpoint.pth
mv decoder_checkpoint.pth ./demo/guitar/bsb
