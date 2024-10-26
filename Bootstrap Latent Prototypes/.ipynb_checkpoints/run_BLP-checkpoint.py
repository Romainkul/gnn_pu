import copy
from torch.optim import AdamW
from blp import *
from load_dataset_and_preprocess import *
import argparse
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='load parameter for running and evaluating \
                                                 boostrap pu learning')
    parser.add_argument('--dataset', '-d', type=str, default='amazon-photos',
                        help='Data set to be used')
    parser.add_argument('--positive_index', '-c', type=int, default=0,
                        help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=1,
                        help='random seed for sample labeled positive from all positive nodes')
    parser.add_argument('--train_pct', '-p', type=float, default=0.2,
                        help='Percentage of positive nodes to be used as training positive')
    parser.add_argument('--val_pct', '-v', type=float, default=0.1,
                        help='Percentage of positive nodes to be used as evaluating positive')
    parser.add_argument('--test_pct', '-t', type=float, default=1.00,
                        help='Percentage of unknown nodes to be used as test set')
    parser.add_argument('--hidden_size', '-l', type=int, default=32,
                        help='Size of hidden layers')
    parser.add_argument('--output_size', '-o', type=int, default=16,
                        help='Dimension of output representations')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pu dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=[args.positive_index], sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct,test_pct=args.test_pct)
    data = data.to(device)
    dataset = [data]

    # prepare augment
    drop_edge_p_1, drop_feat_p_1, drop_edge_p_2, drop_feat_p_2 = agmt_dict[args.dataset]
    augment_1 = augment_graph(drop_edge_p_1, drop_feat_p_1)
    augment_2 = augment_graph(drop_edge_p_2, drop_feat_p_2)

    # build BLP networks
    input_size = data.x.size(1)
    encoder = GCNEncoder(input_size, args.hidden_size, args.output_size)
    predictor = MLP_Predictor(args.output_size, args.hidden_size, args.output_size)
    model = BLP(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=1e-5)
    positive_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_nodes = data.val_mask.nonzero(as_tuple=False).view(-1)

    def learn_repersentations():
        model.train()
        # forward
        optimizer.zero_grad()
        g1, g2 = augment_1(data), augment_2(data)
        p1, aux_h2 = model(g1, g2)
        p2, aux_h1 = model(g2, g1)

        # losses of predicting positive nodes
        positive_loss = predict_positive_nodes(p1, p2, positive_nodes)
        # losses of predicting unlabeled nodes
        unlabeld_loss = predict_unlabeled_nodes(p1, aux_h2.detach(), p2, aux_h1.detach())

        # joint learning
        if unlabeld_loss > positive_loss:
            loss = unlabeld_loss
        else:
            loss = positive_loss

        loss.backward()
        # update main network
        optimizer.step()
        # update axulary network
        model.update_aux_network(0.005)

    def select_reliable_negatives_train_classifier():
        model.eval()
        g1, g2 = augment_1(data), augment_2(data)
        p1, _ = model(g1, g2)
        p2, _ = model(g2, g1)

        if data.train_mask.sum().item() < 5 * positive_nodes.size(0):
            negative_nodes = find_reliable_negative_nodes(p1, p2, positive_nodes, val_nodes)
            # divide negative_nodes into negative train and negative evaluate
            perm_neg_idx = negative_nodes[torch.randperm(negative_nodes.size(0))]
            neg_train_idx = perm_neg_idx[val_nodes.size(0):]  # select negative train
            neg_val_idx = perm_neg_idx[:val_nodes.size(0)]    # select negative evaluate

            data.train_mask[neg_train_idx] = True  # previous train_mask only contains positive nodes
            data.val_mask[neg_val_idx] = True      # previous val_mask only contains positive nodes
            data.y_psd_neg[negative_nodes] = 0     # Assign label 0 to reliable negative nodes

        if data.train_mask.sum().item() >= 5 * positive_nodes.size(0):
            tmp_encoder = copy.deepcopy(model.main_encoder).eval()
            representations = tmp_encoder(data).detach()
            labels = data.y.detach()
            scores = train_binary_classifier_test(representations.cpu().numpy(), data.y_psd_neg.cpu().numpy(),
                                                  labels.cpu().numpy(),
                                                  data.train_mask.cpu().numpy(), data.val_mask.cpu().numpy(), \
                                                  data.test_mask.cpu().numpy())
        else:
            scores = [0.0, 0.0]
        scores.append(data.train_mask.sum().item())
        return scores

    # learn representation, select reliable negatives, and
    data.y_psd_neg = data.y.clone().to(device)  # for recording selecting unlabeled as negative
    for epoch in range(1, 5000):
        learn_repersentations()
        if (epoch % 100 == 0)&(epoch > 3000):
            val_f1_score, test_f1_score, train_num= select_reliable_negatives_train_classifier()
            if data.train_mask.sum().item() >= 5 * positive_nodes.size(0):
                print("epoch: {}, val_f1_score: {:.4f}, test_f1_score: {:.4f}".format(epoch, val_f1_score, test_f1_score))