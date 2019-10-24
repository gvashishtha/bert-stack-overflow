import os
import argparse
import logging
import pandas as pd
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from absl import app
from absl import flags
from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertTokenizer
from transformers.modeling_tf_utils import get_initializer
from azureml.core.run import Run

# Ignore warnings in logs
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Get the Azure ML run object
run = Run.get_context()

# Define input arguments
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length of input sentences.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.', lower_bound=0)
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate for training.')
flags.DEFINE_integer('steps_per_epoch', 150, 'Number of steps per epoch.')
flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train for.', lower_bound=0)
flags.DEFINE_string('data_dir', None, 'Root path of directory where data is stored.')
flags.DEFINE_string('export_dir', './ouputs', 'The directory to export the model to')

class AmlLogger(tf.keras.callbacks.Callback):
    ''' A callback class for logging metrics using Azure Machine Learning Python SDK '''

    def on_epoch_end(self, epoch, logs={}):
        run.log('val_accuracy', float(logs.get('val_accuracy')))

    def on_batch_end(self, batch, logs={}):
        run.log('accuracy', float(logs.get('accuracy')))

class TFBertForMultiClassification(TFBertPreTrainedModel):
    '''BERT Model class for multi-label classification using a softmax output layer '''

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultiClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='softmax')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)

def encode_example(example, tokenizer, max_seq_length, labels_map):
    ''' Encodes an input text using the BERT tokenizer

    :param example: Input line from CSV file
    :param tokenizer: BERT tokenizer object from transformers libary
    :param max_seq_length: Maximum length of word embedding in encoded example
    :param labels_map: Label map dictionary
    :return: Encoded example that can be inputted into the BERT model

    '''
    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        example[1],
        add_special_tokens=True,
        max_length=max_seq_length
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    
    # Get label using dictionary
    label = labels_map[example[2]]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'label': label
    }


def read_csv(filename, tokenizer, max_seq_length, labels_map):
    ''' Reads a CSV file line by line and encodes each example using the BERT tokenizer

    :param filename: The name of the CSV file
    :param tokenizer: BERT tokenizer object from transformers libary
    :param max_seq_length: Maximum length of word embedding in encoded example
    :param labels_map: Label map dictionary
    :return: Encoded examples for BERT model as a generator
    
    '''
    with open(filename, 'r') as f:
        for line in f.readlines():
            record = line.rstrip().split(',')
            features = encode_example(record, tokenizer, max_seq_length, labels_map)
            yield  ({'input_ids': features['input_ids'],
                     'attention_mask': features['attention_mask'],
                     'token_type_ids': features['token_type_ids']},
                      features['label'])

def get_dataset(filename, tokenizer, max_seq_length, labels_map):
    ''' Loads data from a CSV file into a Tensorflow Dataset (while encoding each example)

    :param filename: The name of the CSV file
    :param tokenizer: BERT tokenizer object from transformers libary
    :param max_seq_length: Maximum length of word embedding in encoded example
    :param labels_map: Label map dictionary
    :return: A Tensorflow Dataset object with encoded inputs from CSV file
    
    '''
    generator = lambda: read_csv(filename, tokenizer, max_seq_length, labels_map)
    return tf.data.Dataset.from_generator(generator,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([max_seq_length]),
              'attention_mask': tf.TensorShape([max_seq_length]),
              'token_type_ids': tf.TensorShape([max_seq_length])},
             tf.TensorShape([])))

def main(_):

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Get labels
    labels = pd.read_csv(os.path.join(FLAGS.data_dir,'classes.txt'), header=None)
    labels_map = { row[0]:index for index, row in labels.iterrows() }

    # Load dataset, tokenizer, model from pretrained model/vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForMultiClassification.from_pretrained('bert-base-cased', num_labels=len(labels_map))

    # Load dataset, shuffle data, and put into batchs
    train_dataset = get_dataset(os.path.join(FLAGS.data_dir, 'train.csv'), tokenizer, FLAGS.max_seq_length, labels_map)
    valid_dataset = get_dataset(os.path.join(FLAGS.data_dir, 'valid.csv'), tokenizer, FLAGS.max_seq_length, labels_map)
    test_dataset = get_dataset(os.path.join(FLAGS.data_dir, 'test.csv'), tokenizer, FLAGS.max_seq_length, labels_map)

    # Horovod: multiply batch size by number of gpu's
    train_dataset = train_dataset.shuffle(100).repeat().batch(FLAGS.batch_size * hvd.size())
    valid_dataset = valid_dataset.batch(FLAGS.batch_size * hvd.size())
    test_dataset = test_dataset.batch(FLAGS.batch_size * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate*hvd.size(), epsilon=1e-08, clipnorm=1.0)
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Compile tf.keras model with optimizer, loss, and metric
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric], experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),

        AmlLogger()
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train and evaluate model
    model.fit(train_dataset, epochs=FLAGS.num_epochs, steps_per_epoch=FLAGS.steps_per_epoch // hvd.size(), validation_data=valid_dataset, callbacks=callbacks, verbose=verbose)
    model.evaluate(test_dataset)

    # Export the trained model
    if not os.path.exists(FLAGS.export_dir):
        os.makedirs(FLAGS.export_dir)
    model.save_pretrained(FLAGS.export_dir)

if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    app.run(main)