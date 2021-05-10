from evo import main_seg_ape, main_ape
import argcomplete
import argparse

import logging
import sys

parser = main_ape.parser()
argcomplete.autocomplete(parser)

logger = logging.getLogger(__name__)

def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    merge .json config file with the command line args (if --config was defined)
    :param args: parsed argparse NameSpace object
    :return: merged argparse NameSpace object
    """
    import json
    if args.config:
        with open(args.config) as config:
            merged_config_dict = vars(args).copy()
            # merge both parameter dicts
            merged_config_dict.update(json.loads(config.read()))
            # override args the hacky way
            args = argparse.Namespace(**merged_config_dict)
    return args



args = parser.parse_args()
if hasattr(args, "config"):
    args = merge_config(args)
    
try:
    main_ape.run(args)
except KeyboardInterrupt:
    sys.exit(1)
except SystemExit as e:
    sys.exit(e.code)
except Exception:
    sys.exit(1)
