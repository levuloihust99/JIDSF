import torch

from typing import Optional

from torch import nn
from transformers import (
    BertModel, BertPreTrainedModel,
    RobertaModel, RobertaPreTrainedModel
)


class BertIntentClassifier(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = False, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.add_pooling_layer = add_pooling_layer
        self.bert = BertModel(config, add_pooling_layer=add_pooling_layer)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_encoder(self) -> nn.Module:
        return self.bert
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if self.add_pooling_layer:
            pooled_output = outputs.pooler_output
        else:
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output
