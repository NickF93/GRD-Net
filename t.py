from src.model import BottleNeckType, create_res_ae

def test():
    create_res_ae(bottleneck_type = BottleNeckType.DENSE, initial_padding=10, initial_padding_filters=64)

test()
