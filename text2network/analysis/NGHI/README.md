# Text2Network

## Components

Text2Networks includes four python objects, one for preprocessing, one for training the language model, one for inference and one class that directly represents the semantic network and all operations on it.

The objects correspond to the four steps necessary to analyze a corpus:

1. Pre-processing: Texts are ingested, cleaned and saved in a hdfs database. Associated metadata is created, this notably includes a time variable and a number of freely chosen parameters.

2. Training of the DNN: The researcher determines which combination of parameters constitutes a unit of analysis (for example newspaper-year-issue) and passes this hierarchy to the trainer class, which trains the required number of DNN models. Please see the research paper why this is required to capture linguistic relations in the text corpus correctly.

3. Processing: Using the trained language models, the each word in the corpus is analyzed in its context. For each such occurrence, ties representing probabilistic relations are created. Here, we use a Neo4j graph database to save this network.

4. Analysis: The semantic_network class directly encapsulates the graph database and conditioning operations thereon. Networks created here can either be analyzed directly, or exported in gexf format.

## Prerequisites

To run all elements correctly, the required python packages need to be installed. That is:

Generally: PyTorch (1.3.1+) & Numpy
Preprocessing: tables aka pytables (hdf5) & nltk
Processing: Transformers (2.1.1) & tensorboard
Analysis: networkx & python-louvain & pandas

We are in the process of updating models to newer versions of PyTorch and Transformers.
However, due to some interface changes, the code - and the supplied models - will not run
with the latest versions.

If using anaconda, create a new environment and install

```batch
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch

conda install -c conda-forge transformers=2.1.1

conda install networkx pytables pandas nltk

conda install -c conda-forge tensorboard
```

Finally, a Neo4j server, http accessible, version 4.02+, should be running for processing. We now use the standard BOLT connector, so it is recommended to use the latest version of Neo4j to set up the table.

## Step by Step tutorial

### Configuration

We use the standard python configuration parser to read an ini file.

```python
from text2network.utils.file_helpers import check_create_folder
import configparser

# Load Configuration file
configuration_path = '/config/config.ini'
config = configparser.ConfigParser()
config.read(check_create_folder(configuration_path))
```

Inside the configuration file are a number of fields. It is essential to set the correct paths.

`import_folder` holds the txt data

`pretrained_bert` holds the pre-trained BERT Pytorch model, that is used for all divisions of the corpus.

`trained_berts` will store the fine-tuned BERT models.

`processing_cache` simply keeps track of which subcorpus has already been processed into the Neo4j Graph-

`database` holds the processed text in a hdf5 database.

`log` is the folder for the log.

### Pre-processing

Once text files are comfortably situated in a folder, the text can be pre-processed.
Sentences that are too long are split, tags and other nuisance characters are deleted and so forth.

Most importantly, each sentence is saved in a database, together with its metadata.
This always includes the following:

`Year`: A time variable integer. Typically, YYYY, but YYYYMMDD could be used.

`Source`: Name of the txt file

`p1 through p4`: Up to four parameters coming from the file name

`run_index` An index across all sentences in all text files.

`seq_id` An index across sentences within a given text file.

`text` The sentence, capped at a maximum length of characters.

Since each sentence is then saved as a row in the database, we can determine at a later
stage how we seek to query and split the corpus into subcorpora (e.g. by year and parameter 1).

So initially, we need to use the configuration file to define the properties of the text we are going to use. In particular,
we need to define what the file names mean.
Two options for the file structure are possible:

First, the import folder could include sub-folders of years.

```
    import_folder/
        import_folder/year1/
        ------p1_p2_p3_p4.txt
        ------p1_p2_p3_p4.txt
        (...)
        import_folder/year2/
        ------p1_p2_p3_p4.txt
        ------p1_p2_p3_p4.txt
        (...)
```

Alternatively, all txt files can also reside in a single folder.

```
    import_folder/
        ------year1_p1_p2_p3_p4.txt
        ------year1_p1_p2_p3_p4.txt
        ------year2_p1_p2_p3_p4.txt
        ------year3_p1_p2_p3_p4.txt
        (...)
```

Accordingly, we set the following parameters in the configuration file: `split_symbol`
is the symbol that splits between parameters (here "\_"). `number_params` denotes the number
of parameters (here 4). If we had only two parameters, our text files might be
of the form `p1_p2.txt` and we would set that value to 2.
Finally, `max_seq_length` denotes the maximum length of a sentence.
`char_mult` is a multiplier that determines how many letters the average word can have.
The total sequence length in letters (symbols) is given by `max_seq_length*char_mult`.
Having a fixed-length format here is helpful for performance. Sequences can, of course, be shorter. Later components
will also re-split sentences if smaller batch sizes are desired. Setting the sequence size very high
ensures that no sentence will be unduly split, however this will increase file size.

We begin by instancing the preprocessing class.
At this stage, we will also set up logging.

from src.classes.nw_preprocessor import nw_preprocessor

```python
# Set up preprocessor
preprocessor = nw_preprocessor(config)
```

Note that is is sufficient to pass the `config`, however the class also
takes optional parameters, if we want to overwrite the configuration file.
This is the standard behavior for all modules. So for example one could instead do:

```python
    preprocessor = nw_preprocessor(config, max_seq_length=50)
```

Next, we can process the text files and create the database.
If our text files are split among multiple sub-folders, with years as folder names,
we call the `preprocess_folders` method

```python
preprocessor.preprocess_folders(overwrite=True,excludelist=['checked', 'Error'])
```

here, `overwrite` indicates that we wish to overwrite any existing database.
`excludelist` is a list of strings corresponding to any of the parameters
in the file name. Filenames including elements from this list are not processed.

If, instead, all files are in a single folder, we run

```python
preprocessor.preprocess_files(overwrite=True,excludelist=['checked', 'Error'])
```

Note that both functions also take a `folder` variable, if we want to not use the folder of the configuration file.
In this way, the pre-processing can also be done across many sources. Note, however, that the
file name of the txt file is essential and needs to follow the same convention:
Either folders with year names, or files starting with years, and then up to four parameters.

The module will try to take care of encodings and other matters. If the file can not be read, an error will be
returned.

Once done, a `db.h5` file will be created in the `database` folder, which includes all
individual sentences and their meta-data.

### Training BERT

#### Understanding split hierarchy

We will train one BERT model for each logical division of the corpus. This sub-division will be carried along all subsequent steps. So, processing a certain subdivision requires that a corresponding BERT model has been trained. Different divisions can be trained and saved, as they will be saved in distinct folders.

Subdivisions are specified via the `split_hierarchy` option in the configuration file.

It is a list of parameters by which to split the corpus and train the models. All parameters are always saved as meta-data, but we might want to aggregate across them when training BERT.

The simplest division is by year:

```
    split_hierarchy=["year"]
```

This will train one BERT per year.
However, we might also train one BERT per combination of year, p1 and p2, e.g.

```
    split_hierarchy=["year","p1",p2"]
```

By setting this parameter, the trainer module can ascertain how many BERTs are required, and which sentences it should train on.

#### Training process

We do not wish to use word-pieces. The pre-trained BERT has word-pieces disabled. For that reason, the vocabulary needs to be amended. It is desirable, although not strictly necessary, to use the same vocabulary across all models. To keep this reasonable, set `new_word_cutoff` for large corpora. Only words that occur more often will be included in the vocabulary.

The training process creates first one shared vocabulary, resizes the BERT models and then trains them individually. For this, all texts will be analyzed once to determine what the vocabulary should be. While the system can deal with inconsistent vocabularies by using a translation mechanism, it is highly recommended to have all text data in place.

Each model is trained until either `eval_loss_limit` or `loss_limit` is reached, where the first denotes the loss across test sequences, whereas the second in the current batch during training. The configuration file also includes the usual model parameters, that should be set according to GPU size and corpus size.

To train all BERTs, we initialize the trainer and run the training.
Again, attributes may be given via the config file or as individual parameters.

```python
from text2network.training.bert_trainer import bert_trainer

trainer=bert_trainer(config)
trainer.train_berts()
```

#### Rest omitted as it requires setting up neo4j