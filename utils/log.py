import tensorflow as tf
import logging
import datetime

import tensorflow as tf
import datetime

class FileLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file='training.txt'):
        super(FileLoggingCallback, self).__init__()
        self.log_file = log_file

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.datetime.now()
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1} started at {self.epoch_start_time}\n")

    def on_epoch_end(self, epoch, logs=None):
        # Log information at the end of each epoch
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1} - ")
            f.write(f"Loss: {logs.get('loss'):.4f}, ")
            f.write(f"Accuracy: {logs.get('accuracy'):.4f}, ")
            f.write(f"Val_Loss: {logs.get('val_loss'):.4f}, ")
            f.write(f"Val_Accuracy: {logs.get('val_accuracy'):.4f}\n")

    def on_batch_end(self, batch, logs=None):
        # Optionally, you can log batch-level information here
        pass

# Instantiate the callback
file_logging_callback = FileLoggingCallback()

