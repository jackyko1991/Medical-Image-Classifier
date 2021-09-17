# Experiment Tracking with Guild AI

[Guild AI](https://guild.ai/) is an experiment tracking system for machine learning and allow us to automate grid search for hyperparameters without code change. 

## Installation
Follow installation guide from [official document](https://my.guild.ai/t/get-started-with-guild-ai/35#install-guild-ai).

## Guild AI Pipeline

We have prepared the template configuration json in [./config/config_template.json](/config/config_template.json). Since the original program need to alter configuration file to tune parameters, it is a bit tricky to direct apply Guild AI. Instead, we utilize the pipeline function with three steps:
1. Configuration file generation
2. Training
3. Prediction 

In Guild AI each step will run in independent directory. What you need is to edit [guild.yml](./guild.yml) and change the corresponding hyperparameters under `flags` of `pipeline`:

```yml
pipeline:
    description: Machine learning pipeline for medical image classification
    flags: 
        fold: [1,2,3,4,5]
        epoches: 10
    steps:
    - run: prepare-config
        flags:
            epoches: ${epoches}
```

The `prepare-config` step will use helper file [./config/prepare_config.py](./config/prepare_config.py) to generate the necessary configuraiton json. Currently the helper file is limited to common hyperparameter tuning and you may need to edit to fit your application. To specify the training step to use the generated configuration file, use the command:

```bash
$ guild run pipeline
```

When training is complete use following command to read the output. Since we use Tensorflow to log performance, Guild AI will read all scalars from the `.event` file and creating redundant outputs. 

```bash
$ guild compare
```

Navigate the scalar results with arrow keys, then press `q` to quit.

## Export results and runs
You may export the result with
``` bash
$ guild compare --csv <csv-output-location>
```

Guild runs can be export with 

```bash
guild export [OPTIONS] <output-location> [RUN_ID...]
```
Detail export options can be found from [here]()https://my.guild.ai/t/command-export/80