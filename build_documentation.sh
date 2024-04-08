#!/usr/bin/env sh

rm -rf doc/source
mkdir doc/source
sphinx-apidoc -f -e -o doc/source hog

rm -rf doc/html
sphinx-build -b html doc doc/html
