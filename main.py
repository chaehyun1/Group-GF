import os
import argparse
import torch
from dataset import Dataset  
from model import GroupGF  

def main():
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="CAMRa2011",  # Mafengwo or douban
        help="Either CAMRa2011 or Mafengwo",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Whether to print the results or not. 1 prints the results, 0 does not.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--power", type=float, default=1, help="For normalization of P")
    parser.add_argument("--user_filter", type=int, default=1, help="Filter of user")
    parser.add_argument("--group_filter", type=int, default=2, help="Filter of group")
    parser.add_argument("--uni_filter", type=int, default=3, help="Filter of unified")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k")

    args = parser.parse_args()

    # Print device information if verbose
    if args.verbose:
        print(f"Device: {device}")

    # Load dataset
    current_directory = os.getcwd()
    dataset = args.dataset
    path = os.path.join(current_directory, f'data/{dataset}/')
    data = Dataset(path)
    
    # Retrieve data from the dataset
    R_tr_g, R_ts_g, R_tr_u, g_neg, gu_mat = data.getDataset()

    # Extract dataset statistics
    train_n_groups = R_tr_g.shape[0]
    train_group_n_items = R_tr_g.shape[1]
    train_n_users = R_tr_u.shape[0]
    train_user_n_items = R_tr_u.shape[1]

    if args.verbose:
        print(f"Number of training groups: {train_n_groups}")
        print(f"Number of training group items: {train_group_n_items}")
        print(f"Number of training users: {train_n_users}")
        print(f"Number of training user items: {train_user_n_items}")

    # Initialize and run the GroupGF model
    group_gf = GroupGF(
        args.alpha, args.beta, args.power, args.user_filter,
        args.group_filter, args.uni_filter, args.top_k, device
    )
    hit_result, ndcg_result = group_gf.run_model(
        R_tr_g, R_ts_g, R_tr_u, g_neg, gu_mat,
        train_group_n_items, train_user_n_items
    )

    # Print results
    print(f"NEW MODEL Hit@K: {hit_result:.4f}")
    print(f"NEW MODEL NDCG@K: {ndcg_result:.4f}")

if __name__ == "__main__":
    main()