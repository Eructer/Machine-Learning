import numpy
import time
import re
import math
import json

from net_base import Layer, Network

def main(data_path: str, save_model: bool = False):
    print("Starting")

    reading_time = time.time()

    count = 0

    print("Reading")
    
    final = ""

    with open(data_path, "r", encoding="utf-8") as file:
       
        for line in file:
         
            if count >= 2_000:
          
                break
         
            final += line
          
            count += 1
        
            print(f"Lines read {count}", end="\r")

    print("Text loaded")

    window_size = 2

    embedding_dim = 4

    print("Text processing")

    # Process input sentence
    training_sentence = final.lower()

    training_sentence = re.sub(r"[^\w\s]", "", training_sentence)

    tokens = training_sentence.split()

    vocab = list(set(tokens))

    print(f"Processed Text : Processing time {time.time() - reading_time}")

    print(f"Estimate loss {math.log(len(vocab))}")

    # Setup input for array

    word_to_index = {word: i for i, word in enumerate(vocab)}

    index_to_word = {i: word for word, i in word_to_index.items()}

    vocab_size = len(vocab)

    def generate_skipgram_pairs(tokens: list, window_size: int = 2):
    
        print("Generating pairs")
      
        pairs = []

        for i, center in enumerate(tokens):
         
            for j in range(i - window_size, i + window_size + 1):
              
                if j != i and 0 <= j < len(tokens):
                  
                    pairs.append((center, tokens[j]))
       
        return pairs

    def to_one_hot(word, word_to_index, vocab_size):
        vector = numpy.zeros(vocab_size)

        vector[word_to_index[word]] = 1

        return vector

    # Generate pairs

    pairs = generate_skipgram_pairs(tokens)

    # Make final array inputs for network

    X = []
    Y = []

    for center_word, context_word in pairs:
        X.append(to_one_hot(center_word, word_to_index, vocab_size))
        Y.append(to_one_hot(context_word, word_to_index, vocab_size))

    X = numpy.array(X)
    Y = numpy.array(Y)

    # Setup network

    print("Setting up network")

    epochs = 20

    net = Network()

    net.add_layer(Layer(vocab_size, embedding_dim, activation="none"))
    net.add_layer(Layer(embedding_dim, vocab_size, activation="softmax"))

    # Train

    start_time = time.time()

    print("Starting train cycle")

    losses = net.train(
        X,
        Y,
        epochs=epochs,
        learning_rate=0.01,
        print_loss_every=int(epochs / 8),
    )

    print(f"Final Loss : {losses[len(losses) - 1]}")

    print(f"Train Time : {(time.time() - start_time)/60}")  # Minutes

    # Save model

    if save_model:

        destination = "src/word2vec/model"
        file_name = "word2vec"

        status = net.save_model(
            file_name=file_name,
            destination=destination,
        )
        word_to_index_save_path = f"{destination}/{file_name}_word-to-index.json"
        with open(word_to_index_save_path, "w") as file:
            json.dump(word_to_index, file, indent=4)

        index_to_word_save_path = f"{destination}/{file_name}_index-to-word.json"
        with open(index_to_word_save_path, "w") as file:
            json.dump(index_to_word, file, indent=4)

        if status:
            print(f"Model saved")
        else:
            print(f"Error saving model")

    # Function for getting similarity between vectors
    # def cosine_similarity(v1, v2):
    #     return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))

    # # Function for getting word

    # def vector_to_word(vector, embedding_matrix, index_to_word, top_n=1):
    #     similarities = []

    #     for i, vec in enumerate(embedding_matrix):
    #         sim = cosine_similarity(vector, vec)

    #         similarities.append((i, sim))

    #     similarities.sort(key=lambda x: x[1], reverse=True)

    #     return [index_to_word[i] for i, _ in similarities[:top_n]]


if __name__ == "__main__":
    main("src/word2vec/AllCombined.txt", save_model=True)