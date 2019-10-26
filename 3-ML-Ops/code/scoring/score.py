import os
import json
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


max_seq_length = 128
labels = [
    'azure-web-app-service', 'azure-storage',
    'azure-devops', 'azure-virtual-machine', 'azure-functions'
]


def init():
    global tokenizer, model
    # os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'azure-service-classifier')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'exports')

    # get better debugging output
    try:
        model = TFBertForMultiClassification \
            .from_pretrained(model_dir, num_labels=len(labels))
    except OSError:
        raise OSError(
            'folder actually contained {}'.format(os.listdir(model_dir)))


def run(raw_data):

    # Encode inputs using tokenizer
    inputs = tokenizer.encode_plus(
        json.loads(raw_data)['text'],
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

    # Make prediction
    predictions = model.predict({
        'input_ids': tf.convert_to_tensor([input_ids], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(
            [attention_mask],
            dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor(
            [token_type_ids],
            dtype=tf.int32)
    })

    result = {
        'prediction': str(labels[predictions[0].argmax().item()]),
        'probability': str(predictions[0].max())
    }

    print(result)
    return result
