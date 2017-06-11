class Setting():

    def __init__(self,
        vector_length=200,
        epoch_interval=5,
        decay=0.9,
        norm_clipping=5,
        dropout=0.7,
        gamma=0.5
        ):
        self.vector_length = vector_length
        self.epoch_interval = epoch_interval
        self.decay = decay
        self.norm_clipping = norm_clipping
        self.dropout = dropout
        self.gamma = gamma


class ProposedSetting(Setting):

    def __init__(self, network_size="small", dataset_kind="ptb"):
        super().__init__()
        if network_size == "small":
            self.vector_length = 200
            self.epoch_interval = 5
            self.decay = 0.9
            self.norm_clipping = 5
            self.dropout = 0.7 if dataset_kind == "ptb" else 0.8
        elif network_size == "medium":
            self.vector_length = 650
            self.epoch_interval = 10
            self.decay = 0.9
            self.norm_clipping = 5
            self.dropout = 0.5 if dataset_kind == "ptb" else 0.6
        elif network_size == "large":
            self.vector_length = 1500
            self.epoch_interval = 1
            self.decay = 0.97
            self.norm_clipping = 6
            self.dropout = 0.35 if dataset_kind == "ptb" else 0.6
        
        if dataset_kind == "ptb":
            self.gamma = 0.5
        elif kind == "wiki2":
            self.gamma = 1.0
