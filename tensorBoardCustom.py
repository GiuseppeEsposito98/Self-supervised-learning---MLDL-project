from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import csv

writer = SummaryWriter(log_dir="./results")
def fromcsvToTens(filePath):
    header = ["accuracy","validation_loss", "source_los_cos", "source_los_sin", "target_los_cos", "target_los_sin"]
    df = pd.read_csv(filepath=filePath, usecols=header)
    for i in range(len(df[:,0])):
        for j in range(len(df[0,:])):
            writer.add_scalar("accuracy", df[i,0], i)
            writer.add_scalar("validation_loss", df[i,0], i)
            writer.add_scalar("source_los_cos", df[i,1], i)
            writer.add_scalar("source_los_sin", df[i,2], i)
            writer.add_scalar("target_los_cos", df[i,3], i)
            writer.add_scalar("target_los_sin", df[i,4], i)
            