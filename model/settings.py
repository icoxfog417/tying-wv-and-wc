class SizeSetting():

    @classmethod
    def get(cls, kind):
        if kind == "small":
            return {
                "epoch_interval": 5,
                "decay": 0.9,
                "norm_clipping": 5
            }
        elif kind == "medium":
            return {
                "epoch_interval": 10,
                "decay": 0.9,
                "norm_clipping": 5
            }
        elif kind == "large":
            return {
                "epoch_interval": 1,
                "decay": 0.97,
                "norm_clipping": 6
            }
        else:
            raise Exception("You have to choose size from small, medium, large")            


class DatasetSetting():

    @classmethod
    def get(cls, kind):
        if kind == "ptb":
            return {
                "dropout": {
                    "small": 0.7,
                    "medium": 0.5,
                    "large": 0.35
                },
                "gamma": 0.65
            }
        elif kind == "wiki2":
            return {
                "dropout": {
                    "small": 0.8,
                    "medium": 0.6,
                    "large": 0.6
                },
                "gamma": 1.25
            }
        else:
            raise Exception("You have to choose dataset from ptb, wiki2")            

