# Project Phase 1

## Directory Structure

- Code -> Contains the code
- Outputs -> Sample Inputs and Outputs for tasks 1 and 3.
- Report -> Contains the project report.

### Code

The phase1.py file is a Python3.6+ compatible python script that runs the solution.
The bootstrap.sh file sets up the mongoDb database, and creates the collection

#### Requriements

1. A working Python 3.6+ installation.
2. A working Pipenv installation
3. A working MongoDB installation, running on the default port.

#### Instructions

During the first run, use `./boostrap.sh` to setup the libraries using pipenv and create the mongoDB collections. Alternatively, you can use the following commands to get everything ready.

```bash
mongo mwdb_project --eval 'db.createCollection("image_features")'
pipenv install
```

To execute the various tasks run the code as follows

1. Task 1: `./phase1.py --image_id <ImageID> --model <MODEL NAME - CM/LBP>  <PATH TO FOLDER>  1`
2. Task 2: `./phase1.py <PATH TO FOLDER> 2`
3. Task 3: `./phase1.py --image_id <ImageID> --model <MODEL NAME - CM/LBP> --k <Value of K> <PATH TO FOLDER> 3`
