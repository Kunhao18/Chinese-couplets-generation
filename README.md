# Chinese-couplets-generation

### 1 Directory Structure

- couplet_models: trained model parameters
- client system: client (Qt GUI)
- models.py: definition of the model
- util_funcs.py: utility functions (e.g. data loader)
- run-gen.py demo for generation of couplets
- run-score.py demo for couplets scoring
- run-server.py server launcher

### 2 User Guide

##### 2.1 Couplet Generation

run the `run-gen.py`

##### 2.2 Couplet Scoring

run the `run-score.py`

##### 2.3 Couplet Server

> Since the client is dynamically compiled in Qt, you need to configure the dynamic link library according to the error message

run the `run-server.py` to launch the server, then input “server start”

run the `nlp.exe` to start the client after server launching
