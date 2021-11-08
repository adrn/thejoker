#!/bin/bash -eux

if [[ -z $CIRCLE_PULL_REQUEST ]]; then
    CACHEDIR=`pwd`/theano_cache
    rm -rf $CACHEDIR
    export THEANO_FLAGS=base_compiledir=$CACHEDIR
    export AESARA_FLAGS=base_compiledir=$CACHEDIR

    git branch -D executed-notebooks || true
    git checkout -b executed-notebooks

    . venv/bin/activate
    cd docs
    python run_notebooks.py

    git add examples/*.ipynb
    git -c user.name='circleci' -c user.email='circleci' commit -m "now with executed tutorials"

    git push -f origin executed-notebooks

    echo "Not a pull request: pushing run notebooks branch."
else
    echo $CIRCLE_PULL_REQUEST
    echo "This is a pull request: not pushing executed notebooks."
fi