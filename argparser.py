import sys

# Before parsing arguments just check if Lekhaka is importable !?
try:
    import telugu
except ModuleNotFoundError as e:
    print("\n"
          "Lekhaka is not found in import path.\n"
          "Install it as \n"
          "$pip install -e ../Lekhaka\n"
          "or\n"
          "Install directly from GitHub repo as \n"
          "$pip install -e git+https://github.com/rakeshvar/Lekhaka.git#egg=Lekhaka")
    sys.exit(0)

import argparse

parser = argparse.ArgumentParser(description="Train Chamanti Neural Network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
command = parser.add_subparsers(dest="command", required=True, help="Choose a mode of initialization.")
parser_spec = command.add_parser("spec", help="Init from predefined specs")
parser_spec.add_argument("num", type=int, default=0, help="Number argument for spec")

parser_banti = command.add_parser("banti", help="Init from Banti")
parser_banti.add_argument("pkl_file", type=str, help="Pickle file input")
parser_banti.add_argument("rnnarg", type=str, default="rnn66", help="lstm66 rnn99 etc.")

parser_chamanti = command.add_parser("chamanti", help="Init from chamanti")
parser_chamanti.add_argument("pkl_file", type=str, help="Pickle file input")

parser.add_argument("-O", "--output_dir", type=str, default="models/", help="Directory to save outputs")
parser.add_argument("-Z", "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("-E", "--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-S", "--steps_per_epoch", type=int, default=1000, help="Steps per epoch")

args = parser.parse_args()

