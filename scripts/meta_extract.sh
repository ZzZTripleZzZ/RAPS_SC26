#!/bin/bash

# usage: raps/utils/meta_extract.sh date*.out

for file in "$@"
do
    # Ensure the file exists and is readable
    if [ -r "$file" ]; then
        echo "$file, $(cat "$file" | python raps/utils/meta_extract.py)"
    else
        echo "Cannot read $file"
    fi
done
