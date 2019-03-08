### run this code using

` allennlp train experiments/decomp_attn.json --serialization-dir /tmp/da --recover --include-package models_readers`

- The --recover is if you know you had stopped before in the middle of a good run and want to recover the saved 
checkpoint. Else if you get the error `Serialization directory (/tmp/da) already exists` do `rm -rf /tmp/da/`.  

- `--include-package models_readers` : If running for the first time register the new dataset by adding `--include-package models_readers`


- List of other command line arguments that you can pass can be found [here](https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py)
- Details about how to debug your code (with and without pycharm) can be found [here](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/using_a_debugger.md)
    - If you still would rather work by printing things onto command line you can turn on the DEBUG mode of allennlp using :`export ALLENNLP_DEBUG=True` on your command line.

- To run/debug from pycharm:
    - Run>Edit Configurations>select wrapper_debug.py> apply>OKO
    - don't give any parameters. Its all hardcoded inside wrapper_debug.py. Change there if you want to.
    - Note: don't run actual training using this method. This is because there is a hardcoding inside wrapper_debug.py to remove your serialization directory every time.