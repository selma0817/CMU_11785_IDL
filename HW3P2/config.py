root = '/content/11-785-f24-hw3p2/'

# Feel free to add more items here
config = {
    "beam_width" : 2,
    "lr"         : 2e-3,
    "epochs"     : 50,
    "batch_size" : 64  # Increase if your device can handle it
}

# You may pass this as a parameter to the dataset class above
# This will help modularize your implementation
transforms = [] # set of tranformations