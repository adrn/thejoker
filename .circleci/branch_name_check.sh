#!/bin/bash

if [ -z "${CIRCLE_PULL_REQUEST}" ] && [ "${CIRCLE_BRANCH}" != "main" ] && [[ ! "$CIRCLE_BRANCH" =~ ^v[0-9\.]+ ]];
then
    echo "HALTING"
    circleci step halt
else
    echo "Continuing with tests"
fi

echo $CIRCLE_PULL_REQUEST
echo $CIRCLE_BRANCH
