import os
import argparse
import logging
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from transformers import BertTokenizer
from model import TFBertForMultiClassification
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
flags.DEFINE_string('data_dir', './data', 'Root path of directory where data is stored.')
flags.DEFINE_string('export_dir', './outputs', 'The directory to export the model to')

class AmlLogger(tf.keras.callbacks.Callback):
    ''' A callback class for logging metrics using Azure Machine Learning Python SDK '''

    def on_epoch_end(self, epoch, logs={}):
        run.log('val_accuracy', float(logs.get('val_accuracy')))

    def on_batch_end(self, batch, logs={}):
        run.log('accuracy', float(logs.get('accuracy')))

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

    train_dataset = train_dataset.shuffle(100).repeat().batch(FLAGS.batch_size)
    valid_dataset = valid_dataset.batch(FLAGS.batch_size)
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    # Compile tf.keras model with optimizer, loss, and metric
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # Train and evaluate model
    for item, label in train_dataset:
        with tf.GradientTape() as tape:
            prediction, = model(item, training=True)
            loss_val = loss(label, prediction)
        gradients = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric(label, prediction)
    
    for item, label in test_dataset:
        prediction, = model(item, training=False)
        loss_val = loss(label, prediction)
        metric(label, prediction)

    # Export the trained model
    if not os.path.exists(FLAGS.export_dir):
        os.makedirs(FLAGS.export_dir)
    model.save_pretrained(FLAGS.export_dir)

if __name__ == '__main__':
    app.run(main)