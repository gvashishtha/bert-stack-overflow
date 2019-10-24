import os
import pandas as pd
import argparse
import tensorflow as tf
from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertTokenizer
from transformers.modeling_tf_utils import get_initializer
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class TFBertForMultiClassification(TFBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultiClassification, self) \
            .__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name='classifier',
            activation='softmax')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout(
            pooled_output,
            training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)


def encode_example(example, tokenizer, max_seq_length, labels_map):
    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        example[1],
        add_special_tokens=True,
        max_length=max_seq_length
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens.
    # Only real tokens are attended to.
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
    with open(filename, 'r') as f:
        for line in f.readlines():
            record = line.rstrip().split(',')
            features = encode_example(
                record, tokenizer,
                max_seq_length, labels_map)
            yield ({
                        'input_ids': features['input_ids'],
                        'attention_mask': features['attention_mask'],
                        'token_type_ids': features['token_type_ids']
                    },
                    features['label']  # noqa: E127
                    )  # noqa: E124


def get_dataset(filename, tokenizer, max_seq_length, labels_map):
    generator = lambda: read_csv(  # noqa: E731
        filename, tokenizer,
        max_seq_length, labels_map
    )
    return tf.data.Dataset.from_generator(
        generator,
        (
            {
                'input_ids': tf.int32,
                'attention_mask': tf.int32,
                'token_type_ids': tf.int32
            },
            tf.int64
        ),
        (
            {
                'input_ids': tf.TensorShape([max_seq_length]),
                'attention_mask': tf.TensorShape([max_seq_length]),
                'token_type_ids': tf.TensorShape([max_seq_length])},
            tf.TensorShape([]))
        )


# Create required arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_length',
                    dest='max_seq_length',
                    type=int,
                    help='Maximum sequence length of input sentences.',
                    required=True
                    )
parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    help='Batch size for training.',
                    required=True
                    )
parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    type=float,
                    help='Learning rate for training.',
                    required=True
                    )
parser.add_argument('--steps_per_epoch',
                    dest='steps_per_epoch',
                    type=int,
                    help='Number of steps per epoch.',
                    required=True
                    )
parser.add_argument('--num_epochs',
                    dest='num_epochs',
                    type=int,
                    help='Number of epochs to train for.',
                    required=True
                    )
parser.add_argument('--data_dir',
                    dest='data_dir',
                    help='Root path of directory where data is stored.',
                    required=True
                    )
parser.add_argument('--export_dir',
                    dest='export_dir',
                    help='The directory to export the model to',
                    required=True
                    )
args = parser.parse_args()

# Take in arguments from argparser
max_seq_length = args.max_seq_length
batch_size = args.batch_size
steps_per_epoch = args.steps_per_epoch
learning_rate = args.batch_size
num_epochs = args.num_epochs
data_dir = args.data_dir
export_dir = args.export_dir

# Get labels
labels = pd.read_csv(os.path.join(data_dir, 'classes.txt'), header=None)
labels_map = {row[0]: index for index, row in labels.iterrows()}

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForMultiClassification \
            .from_pretrained('bert-base-cased', num_labels=len(labels_map))

# Load dataset, shuffle data, and put into batchs
train_dataset = get_dataset(os.path.join(data_dir, 'train.csv'),
                            tokenizer,
                            max_seq_length,
                            labels_map
                            )
valid_dataset = get_dataset(os.path.join(data_dir, 'valid.csv'),
                            tokenizer,
                            max_seq_length,
                            labels_map
                            )
test_dataset = get_dataset(os.path.join(data_dir, 'test.csv'),
                           tokenizer,
                           max_seq_length,
                           labels_map
                           )

train_dataset = train_dataset.shuffle(100).repeat().batch(batch_size)
valid_dataset = valid_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Compile tf.keras model with optimizer, loss, and metric
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,
                                     epsilon=1e-08,
                                     clipnorm=1.0
                                     )
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate model
model.fit(train_dataset,
          epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=valid_dataset
          )
model.evaluate(test_dataset)

# Export the trained model
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
model.save_pretrained(export_dir)
