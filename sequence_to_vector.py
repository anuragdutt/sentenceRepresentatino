        # std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    # def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
    #     super(DanSequenceToVector, self).__init__(input_dim)
    #     # TODO(students): start
    #     # ...
    #     self.input_dim = input_dim
    #     self.num_layers = num_layers
    #     self.dropout = dropout

        # TODO(students): end


    # def call(self,
    #          vector_sequence: tf.Tensor,
    #          sequence_mask: tf.Tensor,
    #          training=False) -> tf.Tensor:
    #     # TODO(students): start
    #     # ...

    #     # masked_sequence = tf.boolean_mask(vector_sequence, sequence_mask)
    #     ## Applying Dropout
    #     n_sentences = vector_sequence.get_shape().as_list()[0]
    #     # word_binary = numpy.random.rand(n) > _dropout
    #     combined_list = []
    #     layer_list = []
    #     # print(vector_sequence.get_shape().as_list())
    #     # exit(0)
    #     for s in range(n_sentences):
    #         sen = tf.boolean_mask(vector_sequence[s], sequence_mask[s])
    #         n_words = sen.get_shape().as_list()[0]
    #         retained_list = []
    #         layer_sentence = []
    #         for w in range(n_words):
    #             boolean = numpy.random.rand() > self.dropout
    #             if boolean:
    #                 retained_list.append(sen[w])
    #         retained_words = tf.convert_to_tensor(retained_list)    
    #         vect = tf.reduce_mean(retained_words, axis = 0)            

    #         out = tf.expand_dims(vect, 0)
    #         combined_list.append(vect)


    #         for i in range(self.num_layers):
    #             if i == 0:
    #                 input_d = out.get_shape().as_list()[0]
    #             else:
    #                 input_d = self.input_dim
    #             out = layers.Dense(self.input_dim, input_shape=(input_d,), activation = "relu")(out)
    #             layer_rep = tf.reshape(out, [out.get_shape().as_list()[1]])
    #             layer_sentence.append(layer_rep)

    #         layer_list.append(layer_sentence)

    #     combined_vect = tf.stack(combined_list)
    #     layer_representations = tf.convert_to_tensor(layer_list)

    #     # print(self.num_layers)
    #     # print(combined_vect.get_shape())
    #     # print(layer_representations.get_shape())
    #     # exit(0)

    #     # TODO(students): end
    #     return {"combined_vector": combined_vect,
    #             "layer_representations": layer_representations}

    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hlayers = []
        for i in range(self.num_layers):
            l  = layers.Dense(self.input_dim, activation = 'tanh')
            self.hlayers.append(l)


    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...

        
        ## Applying Dropout
        n_sentences = vector_sequence.get_shape().as_list()[0]
        # word_binary = numpy.random.rand(n) > _dropout
        combined_list = []
        layer_list = []
        # print(vector_sequence.get_shape().as_list())
        # exit(0)
        # masked_sequence = tf.boolean_mask(vector_sequence, sequence_mask)
        # print(sequence_mask.get_shape())
        # print(vector_sequence.get_shape())
        # print(masked_sequence.get_shape())    
        # print()
        # exit(0)

        for s in range(n_sentences):
            sen = tf.boolean_mask(vector_sequence[s], sequence_mask[s])
            n_words = sen.get_shape().as_list()[0]
            dropout_boolean = numpy.random.rand(n_words) > self.dropout
            retained_words = tf.boolean_mask(sen, dropout_boolean)
            retained_count = retained_words.get_shape().as_list()[0] 
            if retained_count > 0:
                vect = tf.reduce_mean(retained_words, axis = 0)           
            else:
                # retained_words = sen
                # retained_count = retained_words.get_shape().as_list()[0]
                # vect = tf.reduce_mean(retained_words, axis = 0)
                
                vect = tf.convert_to_tensor(numpy.zeros(self.input_dim, dtype = numpy.float32))
                # print(vectort)
                # exit(0)

            # retained_list = []
            # layer_sentence = []
            # for w in range(n_words):
            #     boolean = numpy.random.rand() > self.dropout
            #     if boolean:
            #         retained_list.append(sen[w])
            # retained_words = tf.convert_to_tensor(retained_list)    

            out = tf.expand_dims(vect, 0)
            combined_list.append(vect)

        word_reps = tf.stack(combined_list)
        out = word_reps        
        for l in self.hlayers:
            out = l(out)
            layer_list.append(out)     
            # print(out.get_shape())


        #     for i in range(self.num_layers):
        #         if i == 0:
        #             input_d = out.get_shape().as_list()[0]
        #         else:
        #             input_d = self.input_dim
        #         out = layers.Dense(self.input_dim, input_shape=(input_d,), activation = "relu")(out)
        #         layer_rep = tf.reshape(out, [out.get_shape().as_list()[1]])
        #         layer_sentence.append(layer_rep)

        #     layer_list.append(layer_sentence)

        combined_vect = out
        layer_representations_transpose = tf.convert_to_tensor(layer_list)
        layer_representations = tf.transpose(layer_representations_transpose, [1,0,2])
        # print(self.num_layers)
        # print(combined_vect.get_shape())
        # print(layer_representations.get_shape())
        # exit(0)

        # TODO(students): end
        return {"combined_vector": combined_vect,
                "layer_representations": layer_representations}





class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hlayers = []
        for i in range(self.num_layers):
            l  = layers.GRU(self.input_dim, activation = 'tanh', return_sequences = True, return_state = True)
            self.hlayers.append(l)


        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...
        # masked_input = tf.boolean_mask(vector_sequence, sequence_mask)
        

        # for i in range(self.num_layers):
        #     if i == 0:
        #         # input_d = vector_sequence.get_shape().as_list()
        #         l  = layers.GRU(self.input_dim, activation = 'tanh', return_sequences = False)
        #     else:
        #         # input_d = [vector_sequence.get_shape().as_list()[0], self.input_dim, vector_sequence.get_shape().as_list()[0]]
        #         l = layers.GRU(self.input_dim, activation = 'tanh', return_sequences = False)
        #         # l = layers.Dense(self.input_dim, activation = 'tanh')
        #     hlayers.append(l)

        # inp = vector_sequence
        layer_list = []
        for j in range(len(self.hlayers)):
            h = self.hlayers[j]
            if j == 0:
                inp_all = h(vector_sequence, mask = sequence_mask)
                inp = inp_all[0]
                lr = inp_all[1]
            else:
                inp_all = h(inp, mask = sequence_mask)
                inp = inp_all[0]
                lr = inp_all[1]

            # print(inp.get_shape())
            # print(lr.get_shape())

            layer_list.append(lr)
            
            # print(self.input_dim)
            # print(inp.get_shape())
            # exit(0)

        combined_vector = lr
        layer_representations_transpose = tf.convert_to_tensor(layer_list)
        # print(combined_vector.get_shape())
        # print(layer_representations_transpose.get_shape())
        # exit(0)        
        layer_representations = tf.transpose(layer_representations_transpose, [1,0,2])
        # print(layer_representations.get_shape())
        # exit(0)

            # else:
            #     input_d = [vector_sequence.get_shape().as_list()[0],
            #                 input_dim,
            #                 vector_sequence.get_shape().as_list()[2]]

        # for i in range(self.num_layers):
            # layers.GRU(self.)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
