from Censor_SGD_ANN_lib import *

# SGD
config = SGDConfig.copy()
run(optimizer = SGD, config = config, device=device)

# LAG_S
config = LAGSConfig.copy()
run(optimizer = LAG_S, config = config, device=device)

# CSGD
config = CSGDConfig.copy()
run(optimizer = CSGD, config = config, device=device)
