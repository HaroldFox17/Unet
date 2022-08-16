from model import *
from dataset import train_dataset, val_dataset
"""Create instance of the model, then train and save it"""

unet = UNet(768, 768, 3)
model = unet.create_model()
model.fit(train_dataset, validation_data=val_dataset, verbose=2, epochs=3)

model.save("saved_model/unet")
