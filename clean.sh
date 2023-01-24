#!/bin/bash

# Remove created folders
for folder in `find . | grep -E "(__pycache__$)"`
do
	rm -rf $folder
done

# Remove shared library files
for file in `find . -name '*.so'`
do
	rm $file
done

# Remove C files
for file in `find . -name '*.c'`
do
	rm $file
done

# Remove build folder
rm -rf build