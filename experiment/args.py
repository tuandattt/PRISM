import datetime
import torch
import argparse
import os


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--epochs_validate', type=int, default=1, help='model validate epoch interval')
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 help='cuda:[d] | cpu')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to reproduce performance/dataset')
        self.parser.add_argument('--dataset', type=str, default='mmlu',
                        help='the name of the dataset. Default: caveman_small')
        self.parser.add_argument('--dataset_name', default='',
                                 help='Name of the dataset (e.g., mmlu)')
        self.parser.add_argument('--data_dir', type=str, default='../../ColdStartData',
                        help='Data directory for storing dataset files, such as MMLU graph data')
        self.parser.add_argument('--hidden_size_node_level_transformer', type=int, default=256,
                                 help='hidden size for node level transformer')
        self.parser.add_argument('--embedding_size_node_level_transformer', type=int, default=256,
                                 help='the size for node level transformer input')
        self.parser.add_argument('--embedding_size_node_output', type=int, default=256,
                                 help='the size of node output embedding')
        self.parser.add_argument('--hidden_size_edge_level_transformer', type=int, default=256,
                                 help='hidden size for edge level transformer')
        self.parser.add_argument('--embedding_size_edge_level_transformer', type=int, default=256,
                                 help='the size for edge level transformer input')

        self.parser.add_argument('--max_prev_node', type=int, default=10)
        self.parser.add_argument('--max_head_and_tail', type=int, default=10)
        self.parser.add_argument('--is_dag', action='store_true', help='Process directed acyclic graphs')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batchsize')
        self.parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')
        self.parser.add_argument('--epochs', type=int, default=51, help='epochs')

        self.parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.3, help='Learning rate decay factor')
        self.parser.add_argument('--clip', default=True, action='store_true',
                                 help='whether to use clip gradient for generation model')
        self.parser.add_argument('--role_mapping_path', type=str, default=None,
                        help='Path to predefined role mapping file, if provided will load role mapping from here')

        # ------------------------------------------------------------------ #
        # OPSD: on-policy self-distillation from a privileged teacher.
        # ------------------------------------------------------------------ #
        self.parser.add_argument('--alpha', type=float, default=0.2,
                                 help='Weight of the node term in Eq. 19 (edges get 1 - alpha)')
        self.parser.add_argument('--embedding_dim', type=int, default=384,
                                 help='Sentence-encoder dimension (all-MiniLM-L6-v2 = 384)')
        self.parser.add_argument('--legacy_edge_polarity', action='store_true',
                                 help='Reproduce the published sampler, which reads the '
                                      'no-edge channel and so draws edges with P(no edge)')

        # -- teacher (stage 1) --
        self.parser.add_argument('--teacher_epochs', type=int, default=40)
        self.parser.add_argument('--teacher_lr', type=float, default=1e-4)
        self.parser.add_argument('--privileged_only', action='store_true',
                                 help='Train only the privileged channel, freezing the shared '
                                      'trunk: isolates what the rationale adds from more training')
        self.parser.add_argument('--privileged_dropout', type=float, default=0.0,
                                 help='Probability of hiding the rationale during teacher '
                                      'training (>0 gives the Baseline-0 privileged-dropout model)')
        self.parser.add_argument('--privileged_dropout_p', type=float, default=0.1,
                                 help='Dropout inside the answer encoder')
        self.parser.add_argument('--privileged_rank', type=int, default=16,
                                 help='Rank of the privileged bottleneck. Cold start yields '
                                      'O(100) graphs, so a full-width channel is thousands of '
                                      'parameters per example and memorises instead of learning. '
                                      '0 = full width (the ablation).')

        # -- distillation (stage 2) --
        self.parser.add_argument('--opsd_epochs', type=int, default=30)
        self.parser.add_argument('--opsd_lr', type=float, default=1e-4)
        self.parser.add_argument('--opsd_batch_size', type=int, default=16,
                                 help='Queries per step (rollouts per step = this x rollouts_per_query)')
        self.parser.add_argument('--rollouts_per_query', type=int, default=4,
                                 help='Pseudo-graphs sampled per query per step')
        self.parser.add_argument('--rollout_temperature', type=float, default=1.0,
                                 help='Sampling temperature for pseudo-graphs; >1 widens the '
                                      'state distribution the teacher gets to correct')
        self.parser.add_argument('--kl_direction', type=str, default='reverse',
                                 choices=['reverse', 'forward', 'jsd'])
        self.parser.add_argument('--teacher_temperature', type=float, default=1.0,
                                 help='Softens the teacher target; the student can never see the '
                                      'rationale, so it should be allowed more entropy')
        self.parser.add_argument('--jsd_beta', type=float, default=0.5)
        self.parser.add_argument('--anchor_weight', type=float, default=0.3,
                                 help='Weight of the ground-truth NLL anchor (absorbs D_replay)')
        self.parser.add_argument('--feed_edge_features', type=int, default=1,
                                 help='Feed f_edge at each node step, as training does; 0 '
                                      'reproduces the published sampler, which feeds zeros')
        self.parser.add_argument('--pseudo_filter', type=str, default='nonempty',
                                 choices=['all', 'nonempty', 'teacher'],
                                 help="'teacher' keeps the top fraction by teacher "
                                      'log-likelihood -- the LLM-free stand-in for S(Q,G)')
        self.parser.add_argument('--teacher_keep_frac', type=float, default=0.5)
        self.parser.add_argument('--opsd_query_limit', type=int, default=None,
                                 help='Cap the rollout query pool (default: all available)')

    def update_args_from_dict(self, args_dict):
        """
        Update parameters from dictionary.
        """
        for key, value in args_dict.items():
            setattr(self, key, value)

    def update_args(self):
        """
        Update args when load a trained model: use settings from the saved model
        """
        args, _ = self.parser.parse_known_args()
        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
        args.is_dag = True
        args.dir_input = 'output/'
        args.dataset_path = 'datasets/'
        args.dataset_name = args.dataset
        if args.role_mapping_path is not None and os.path.exists(args.role_mapping_path):
            try:
                import json
                with open(args.role_mapping_path, 'r') as f:
                    args.role_mapping = json.load(f)
                print(f"Loaded {len(args.role_mapping)} role mappings from {args.role_mapping_path}")
            except Exception as e:
                print(f"Failed to load role mapping: {e}")
        return args
