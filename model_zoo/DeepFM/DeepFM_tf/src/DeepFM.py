import tensorflow as tf
from yactr.TFscr.models import BaseModel
from yactr.TFscr.layers import FeatureEmbedding, MLP_Block, FactorizationMachine


class DeepFM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DeepFM",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, model_id=model_id, **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim, embedding_regularizer=embedding_regularizer)
        self.fm = FactorizationMachine(feature_map, regularizer=embedding_regularizer)
        self.emb_out_dim = feature_map.sum_emb_out_dim()
        self.mlp = MLP_Block(input_dim=self.emb_out_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm,
                             regularizer=net_regularizer)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)


    def call(self, inputs, training=False):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(tf.reshape(feature_emb, (-1, self.emb_out_dim)))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict