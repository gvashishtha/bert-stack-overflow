import tensorflow as tf
from transformers import TFBertPreTrainedModel, TFBertMainLayer
from transformers.modeling_tf_utils import get_initializer

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