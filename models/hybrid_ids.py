from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, Input

def build_hybrid_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        
        # Two BiGRU layers with 200 and 100 neurons [cite: 198]
        Bidirectional(GRU(200, return_sequences=True)),
        Dropout(0.003), # 0.3% dropout rate [cite: 198]
        Bidirectional(GRU(100, return_sequences=True)),
        Dropout(0.003),
        
        # Two LSTM layers with 100 and 50 neurons [cite: 198]
        LSTM(100, return_sequences=True),
        Dropout(0.003),
        LSTM(50),
        Dropout(0.003),
        
        # Softmax for multi-classification [cite: 199, 201]
        Dense(num_classes, activation='softmax')
    ])
    
    # Categorical cross-entropy loss and ADAM optimizer [cite: 199, 200]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model