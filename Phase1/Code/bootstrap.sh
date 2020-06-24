#!/bin/bash
#brew tap mongodb/brew
#brew install mongodb-community@4.2
mongo mwdb_project --eval 'db.createCollection("image_features")'
pipenv install
