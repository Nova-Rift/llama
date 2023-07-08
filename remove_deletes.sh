#!/bin/bash

# Deleting files
for file in /home/clark/pytorch/language_models/llama/delete*
do
  if [ -f "$file" ]; then
    echo "Deleting $file"
    rm "$file"
  fi
done

