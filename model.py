import torch
import torch.nn as nn
import torchvision.models as models

# Encoder CNN architecture.
class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # Batch normalization
        self.batchnormalization = nn.BatchNorm1d(embed_size)


    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchnormalization(features)
        return features


# Decoder RNN architecture.
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob=0.2):

        super(DecoderRNN, self).__init__()

        # sizes of the model's blocks
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Adding Dropout Layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # Lstm unit(s)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first = True, dropout = 0.5, num_layers = self.num_layers)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        
        # Set bias tensor to all 0.01
        self.fc_out.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(fc_out.weight)
        
        # Adding bias to lstm improves the performance
        
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1]

        # embed the captions
        captions_embed = self.embed(captions)

        # Concatenate the features and caption embeddings.
        input_vals = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        
        # pass to LSTM layer.
        outputs, hidden = self.lstm(input_vals)

        #pass through dropout layer
        outputs = self.dropout(outputs)
        
        # pass through the linear unit
        outputs = self.fc_out(outputs)

        return outputs

    def sample(self, inputs, states=None, max_len=20):

        outputs = []   
        output_length = 0
        predicted_index = 0
        
        while (output_length != max_len+1) and (predicted_index != 1):
            
            ''' LSTM layer '''
            
            output, states = self.lstm(inputs,states)
            
            ''' Linear layer '''
            
            output = self.fc_out(output.squeeze(dim = 1))
            print("Output from FC")
            
            _, predicted_index = torch.max(output, 1)
            
            print("Predicted Index")
            print(predicted_index.cpu().numpy()[0].item())
            
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            inputs = self.embed(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            # To move to the next iteration of the while loop.
            output_length += 1

        return outputs
