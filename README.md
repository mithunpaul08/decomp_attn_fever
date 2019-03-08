### run this code using

` allennlp train experiments/decomp_attn.json --serialization-dir /tmp/da `

if you get error `Serialization directory (/tmp/da) already exists` do `rm -rf /tmp/da/`. Unless ofcourse you 
know you had stopped before in the middle of a good run. In that case add `--recover` to the command above.

If running for the first time register by adding `--include-package models_readers`