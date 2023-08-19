class SentencePieceTokenizer:
    def __init__(self, sp_model):
        self._sp_model = sp_model
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'

    def tokenize(self, text):
        return self._sp_model.encode_as_pieces(text)

    def encode(self, text):
        return self._sp_model.encode_as_ids(text)

    def convert_tokens_to_ids(self, tokens):
        return [self._sp_model.PieceToId(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._sp_model.IdToPiece(id) for id in ids]

    def convert_token_to_id(self, token):
        return self._sp_model.PieceToId(token)

    def convert_id_to_token(self, id):
        return self._sp_model.IdToPiece(id)

    @property
    def unk_token_id(self):
        return self._sp_model.PieceToId(self.unk_token)

    @property
    def pad_token_id(self):
        return self._sp_model.PieceToId(self.pad_token)

    @property
    def sep_token_id(self):
        return self._sp_model.PieceToId(self.sep_token)
    
    @property
    def cls_token_id(self):
        return self._sp_model.PieceToId(self.cls_token)
    
    @property
    def mask_token_id(self):
        return self._sp_model.PieceToId(self.mask_token)