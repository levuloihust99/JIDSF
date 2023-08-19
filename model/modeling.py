import torch
from torch import nn
from transformers import (
    RobertaForTokenClassification, BertPreTrainedModel,
    BertModel, ElectraPreTrainedModel,
    ElectraModel, RobertaModel, XLMRobertaForTokenClassification,
    DistilBertPreTrainedModel, DistilBertModel
)
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
from crf import CRF
from sadice import SelfAdjDiceLoss


class BertPosTagger(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertPosTagger, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.args = args
        self.dropout = nn.Dropout(self.args.dropout_prob)
        if args.pool_type == 'concat':
            self.fc = nn.Linear(int(self.args.num_hidden_layer) * config.hidden_size, self.num_labels)
        else:
            self.fc = nn.Linear(config.hidden_size, self.num_labels)

        self.store_dict = args.__dict__

        if self.args.use_crf:
            self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None
    ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            return_dict=True)

        if self.args.pool_type == "concat":
            outputs = torch.cat(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=-1)
        else:
            outputs = torch.mean(torch.stack(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=0), dim=0)
        embedded = self.dropout(outputs)
        logits = self.fc(embedded)

        outputs = (logits,)

        loss = None
        if labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                outputs = (slot_loss,) + outputs
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.args.ignore_index)
                loss_unbalance = SelfAdjDiceLoss(reduction="mean")
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertPosTagger(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super(DistilBertPosTagger, self).__init__(config)
        self.num_labels = config.num_labels
        self.args = args
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        if args.pool_type == 'concat':
            self.fc = nn.Linear(int(self.args.num_hidden_layer) * config.hidden_size, self.num_labels)
        else:
            self.fc = nn.Linear(config.hidden_size, self.num_labels)

        self.store_dict = args.__dict__

        if self.args.use_crf:
            self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if self.args.pool_type == "concat":
            outputs = torch.cat(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=-1)
        else:
            outputs = torch.mean(torch.stack(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=0), dim=0)
        embedded = self.dropout(outputs)
        logits = self.fc(embedded)

        outputs = (logits,)

        loss = None
        if labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                outputs = (slot_loss,) + outputs
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.args.ignore_index)
                loss_unbalance = SelfAdjDiceLoss(reduction="mean")
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertPosTaggerElectra(ElectraPreTrainedModel):
    def __init__(self, config, args):
        super(BertPosTaggerElectra, self).__init__(config)
        self.electra = ElectraModel(config)
        self.num_labels = config.num_labels
        self.args = args
        self.store_dict = args.__dict__
        self.dropout = nn.Dropout(self.args.dropout_prob)
        if args.pool_type == 'concat':
            self.fc = nn.Linear(int(self.args.num_hidden_layer) * config.hidden_size, self.num_labels)
        else:
            self.fc = nn.Linear(config.hidden_size, self.num_labels)

        if self.args.use_crf:
            self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               return_dict=True)

        if self.args.pool_type == "concat":
            outputs = torch.cat(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=-1)
        else:
            outputs = torch.mean(torch.stack(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=0), dim=0)

        embedded = self.dropout(outputs)
        logits = self.fc(embedded)

        outputs = (logits,)

        loss = None
        if labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                outputs = (slot_loss,) + outputs
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.args.ignore_index)
                loss_unbalance = SelfAdjDiceLoss(reduction="mean")

                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class PhoBertPosTagger(RobertaForTokenClassification):
    def __init__(self, config, args):
        super(PhoBertPosTagger, self).__init__(config)
        self.num_labels = config.num_labels
        if args.pool_type == 'concat':
            self.classifier = nn.Linear(int(args.num_hidden_layer) * config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.args = args
        self.store_dict = args.__dict__
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        if self.args.pool_type == "concat":
            sequence_output = torch.cat(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=-1)
        else:
            sequence_output = torch.mean(torch.stack(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=0), dim=0)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)

        loss = None
        if labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                outputs = (slot_loss,) + outputs
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.args.ignore_index)
                loss_unbalance = SelfAdjDiceLoss(reduction="mean")
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs


class XLMRobertaPosTagger(XLMRobertaForTokenClassification):
    def __init__(self, config, args):
        super(XLMRobertaPosTagger, self).__init__(config)
        self.num_labels = config.num_labels
        if args.pool_type == 'concat':
            self.classifier = nn.Linear(int(args.num_hidden_layer) * config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.args = args
        self.store_dict = args.__dict__
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        if self.args.pool_type == "concat":
            sequence_output = torch.cat(outputs.hidden_states[-int(self.args.num_hidden_layer):], dim=-1)
        else:
            sequence_output = torch.mean(torch.stack(outputs[2][-int(self.args.num_hidden_layer):], dim=0), dim=0)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.args.use_crf:
                slot_loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
                outputs = (slot_loss,) + outputs
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.args.ignore_index)
                loss_unbalance = SelfAdjDiceLoss(reduction="mean")
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    if self.args.use_dice_loss:
                        loss += loss_unbalance(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

        return outputs
