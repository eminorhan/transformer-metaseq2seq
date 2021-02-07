import argparse
import time
import torch
import torch.nn as nn
import generate_episode as ge
from generate_data import Lang, tabu_update, get_unique_words, generate_prim_permutation, generate_batch
from models import TransformerModel

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

    # create validation episodes; these are fixed throughout training, so we need to generate only once
    tabu_episodes = set([])
    samples_val = []
    for i in range(args.num_episodes_val):
        sample = generate_episode_test(tabu_episodes)
        samples_val.append(sample)
        tabu_episodes = tabu_update(tabu_episodes, sample['identifier'])

    x_val = torch.full((400, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol
    y_val = torch.full((980, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol  
    z_val = torch.full((980, len(samples_val)), 19, dtype=torch.int64)  # pad with SOS symbol  

    for i in range(len(samples_val)):
        x_b = torch.cat((samples_val[i]['xs_padded'].flatten(), samples_val[i]['xq_padded'].flatten()))
        y_s = samples_val[i]['ys_padded'].flatten()
        y_q = samples_val[i]['yq_padded'].flatten()

        x_val[:len(x_b), i] = x_b
        y_val[:len(y_s), i] = y_s
        z_val[:len(y_q), i] = y_q

    # start training
    episode_start_time = time.time()
    for episode in range(1, args.num_episodes+1):

        model.train()

        # generate batch
        x, y, z = generate_batch(args.batch_size, generate_episode_train, tabu_episodes)

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
                val_out = model(x_val.cuda(), y_val.cuda(), src_mask=None, tgt_mask=None)
                val_loss = criterion(val_out.view(-1, ntoken), z_val.view(-1).cuda())
                # val_out_top = torch.argmax(val_out, dim=-1)

                print('-' * 89)
                print('| iteration {:6d} | time/iter: {:5.2f}s | validation loss {:5.4f} | '.format(episode, (time.time() - episode_start_time) / args.eval_interval, val_loss.item()))
                print('-' * 89)

                episode_start_time = time.time()