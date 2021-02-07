import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import generate_episode as ge
from models import TransformerModel
from copy import deepcopy, copy

# CONSTANTS
USE_CUDA = torch.cuda.is_available()
SOS_token = "SOS"  # start of sentence
EOS_token = "EOS"  # end of sentence
PAD_token = SOS_token  # padding symbol
USE_RECONSTRUCT_LOSS = False  # whether support items are included also as query items
MAX_TRY_NOVEL = 100  # number of attempts to find a novel episode (not in tabu list) before throwing an error

class Lang:
    # Class for converting strings/words to numerical indices, and vice versa.
    # Should use separate class for input language (English) and output language (actions)
    
    def __init__(self, symbols):
        # symbols : list of all possible symbols
        n = len(symbols)
        self.symbols = symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)

    def variableFromSymbols(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        # 
        # Input
        #  mylist : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices)
        if USE_CUDA:
            output = output.cuda()
        return output

    def symbolsFromVector(self, v):
        # Convert indices to symbols, breaking where we get a EOS token
        # 
        # Input
        #  v : list of m indices
        #   
        # Output
        #  mylist : list of m or m-1 symbols (excluding EOS)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

def make_hashable(G):
    # Separate and sort stings, to make unique string identifier for an episode
    #
    # Input
    #   G : string of elements separate by \n, specifying the structure of an episode 
    G_str = str(G).split('\n')
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def tabu_update(tabu_list, identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier, (list, set)):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier, str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list

def get_unique_words(sentences):
    # Get a list of all the unique words in a list of sentences
    # 
    # Input
    #  sentences: list of sentence strings
    # Output
    #   words : list of all unique words in sentences
    words = []
    for s in sentences:
        for w in s.split(' '): # words
            if w not in words:
                words.append(w)
    return words

def pad_seq(seq, max_length):
    # Pad sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def build_padded_var(list_seq, lang):
    # Transform python list to a padded torch tensor
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = [z+[EOS_token] for z in list_seq]
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths)
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.variableFromSymbols(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded,dim=0)
    if USE_CUDA:
        z_padded = z_padded.cuda()
    return z_padded, z_lengths

def build_sample(x_support, y_support, x_query, y_query, input_lang, output_lang, myhash, grammar=''):
    # Build an episode from input/output examples
    # 
    # Input
    #  x_support [length ns list of lists] : input sequences (each a python list of words/symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
    #  input_lang: Language object for input sequences (see Language)
    #  output_lang: Language object for output sequences
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  grammar : (optional) grammar object
    #
    # Output
    #  sample : dict that stores episode information
    sample = {}

    # store input and output sequences
    sample['identifier'] = myhash
    sample['xs'] = x_support 
    sample['ys'] = y_support
    sample['xq'] = x_query
    sample['yq'] = y_query
    sample['grammar'] = grammar
    
    # convert strings to indices, pad, and create tensors ready for input to network
    sample['xs_padded'], sample['xs_lengths'] = build_padded_var(x_support, input_lang)  # (ns x max_length)
    sample['ys_padded'], sample['ys_lengths'] = build_padded_var(y_support, output_lang)  # (ns x max_length)
    sample['xq_padded'], sample['xq_lengths'] = build_padded_var(x_query, input_lang)  # (nq x max_length)
    sample['yq_padded'], sample['yq_lengths'] = build_padded_var(y_query, output_lang)  # (nq x max_length)
    
    return sample

def generate_prim_permutation(shuffle, nsupport, nquery, input_lang, output_lang, scan_var_tuples, nextra, tabu_list=[]):
    # Generate a SCAN episode with primitive permutation.
    #  The tabu list identifier is based on the permutation of primitive inputs to primitive actions.
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output sequences with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities

    count = 0
    while True:
        D_support, D_query, D_primitive = ge.sample_augment_scan(nsupport, nquery, scan_var_tuples, shuffle, nextra, inc_support_in_query=USE_RECONSTRUCT_LOSS)
        D_str = '\n'.join([s[0] + ' -> ' + s[1] for s in D_primitive])
        identifier = make_hashable(D_str)
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
        count += 1
        if count > MAX_TRY_NOVEL:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]

    return build_sample(x_support, y_support, x_query, y_query, input_lang, output_lang, identifier)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta-seq2seq with generic transformers')
    parser.add_argument('--num_episodes_val', type=int, default=5, help='number of episodes to use as validation throughout learning')
    parser.add_argument('--num_episodes', type=int, default=999999, help='number of training episodes')
    parser.add_argument('--nlayers', type=int, default=4, help='number of layers')
    parser.add_argument('--emsize', type=int, default=512, help='embedding size')
    parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--nhid', type=int, default=1024, help='the number of hidden units in the feedforward layers of the transformer model')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--eval-interval', type=int, default=1000, help='evaluate model at this rate')

    args = parser.parse_args()
    print(args)

    # Set the random seed manually for better reproducibility
    torch.manual_seed(args.seed)

    scan_all = ge.load_scan_file('all', 'train')
    scan_all_var = ge.load_scan_var('all', 'train')

    input_symbols_scan = get_unique_words([c[0] for c in scan_all])
    output_symbols_scan = get_unique_words([c[1] for c in scan_all])

    all_symbols_scan = input_symbols_scan + output_symbols_scan
    all_lang = Lang(all_symbols_scan)
    ntoken = all_lang.n_symbols

    # set up transformer encoder-decoder model, loss, optimizer
    model = TransformerModel(ntoken=ntoken, emsize=args.emsize, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    generate_episode_train = lambda tabu_episodes : generate_prim_permutation(shuffle=True, 
                                                                              nsupport=20, 
                                                                              nquery=20, 
                                                                              input_lang=all_lang, 
                                                                              output_lang=all_lang, 
                                                                              scan_var_tuples=scan_all_var, 
                                                                              nextra=0, 
                                                                              tabu_list=tabu_episodes)

    generate_episode_test = lambda tabu_episodes : generate_prim_permutation(shuffle=False, 
                                                                             nsupport=20, 
                                                                             nquery=20, 
                                                                             input_lang=all_lang, 
                                                                             output_lang=all_lang, 
                                                                             scan_var_tuples=scan_all_var, 
                                                                             nextra=0, 
                                                                             tabu_list=tabu_episodes)

    # create validation episodes
    tabu_episodes = set([])
    samples_val = []
    for i in range(args.num_episodes_val):
        sample = generate_episode_test(tabu_episodes)
        samples_val.append(sample)
        tabu_episodes = tabu_update(tabu_episodes, sample['identifier'])

    episode_start_time = time.time()

    # start training
    for episode in range(1, args.num_episodes+1):

        model.train()

        # Generate a batch
        x = torch.full((400, args.batch_size), 19, dtype=torch.int64)  # pad with SOS symbol
        y = torch.full((980, args.batch_size), 19, dtype=torch.int64)  # pad with SOS symbol  
        z = torch.full((980, args.batch_size), 19, dtype=torch.int64)  # pad with SOS symbol  

        for i in range(args.batch_size):
            sample = generate_episode_train(tabu_episodes)

            x_b = torch.cat((sample['xs_padded'].flatten(), sample['xq_padded'].flatten()))
            y_s = sample['ys_padded'].flatten()
            y_q = sample['yq_padded'].flatten()

            x[:len(x_b), i] = x_b
            y[:len(y_s), i] = y_s
            z[:len(y_q), i] = y_q

        # update params
        model.zero_grad()
        output = model(x.cuda(), y.cuda(), src_mask=None, tgt_mask=None)
        loss = criterion(output.view(-1, ntoken), z.view(-1).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % args.eval_interval == 0:

            model.eval()

            with torch.no_grad():
                # Generate a batch
                x = torch.full((400, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol
                y = torch.full((980, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol  
                z = torch.full((980, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol  

                for i in range(len(samples_val)):
                    x_b = torch.cat((samples_val[i]['xs_padded'].flatten(), samples_val[i]['xq_padded'].flatten()))
                    y_s = samples_val[i]['ys_padded'].flatten()
                    y_q = samples_val[i]['yq_padded'].flatten()

                    x[:len(x_b), i] = x_b
                    y[:len(y_s), i] = y_s
                    z[:len(y_q), i] = y_q

                val_out = model(x.cuda(), y.cuda(), src_mask=None, tgt_mask=None)
                val_loss = criterion(val_out.view(-1, ntoken), z.view(-1).cuda())
                # eval_out = torch.argmax(eval_out, dim=-1)

                print('-' * 89)
                print('| iteration {:6d} | time/iter: {:5.2f}s | validation loss {:5.4f} | '.format(episode, (time.time() - episode_start_time) / args.eval_interval, val_loss.item()))
                print('-' * 89)

                episode_start_time = time.time()