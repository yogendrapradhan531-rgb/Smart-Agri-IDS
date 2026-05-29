import tensorflow as tf
import numpy as np

class TBPTTTrainer:
    """
    Implements Truncated Backpropagation Through Time (TBPTT)
    to handle lengthy sequences of SA data for faster learning.
    """
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @tf.function
    def train_step(self, x_chunk, y_chunk):
        with tf.GradientTape() as tape:
            # Forward process: calculate hidden states
            predictions = self.model(x_chunk, training=True)
            # Calculate Categorical Cross-Entropy Loss
            loss = self.loss_fn(y_chunk, predictions)
        
        # Backpropagation through the truncated length L < T
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def train_on_sequences(self, dataset, sequence_length=10):
        """
        Segments the data into smaller windows to eliminate the need
        of maintaining the entire history.
        """
        total_loss = 0
        steps = 0
        
        for x_batch, y_batch in dataset:
            # Truncate sequence into smaller time-steps
            # Example: splitting a long sequence into windows of 10 steps
            num_chunks = x_batch.shape[1] // sequence_length
            
            for i in range(num_chunks):
                start = i * sequence_length
                end = start + sequence_length
                
                x_chunk = x_batch[:, start:end, :]
                y_chunk = y_batch[:, start:end, :]
                
                loss = self.train_step(x_chunk, y_chunk)
                total_loss += loss
                steps += 1
                
        return total_loss / steps

# Example Usage
# Optimizer: ADAM as per Section III-C
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# trainer = TBPTTTrainer(my_hybrid_model, loss_fn, optimizer)
# trainer.train_on_sequences(my_iot_dataset, sequence_length=10)