import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from psinet.simulation.simulator import Simulator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run a PSINet simulation from a config file.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    sim = Simulator(config_path=args.config_file)
    sim.build()
    sim.run()
    sim.save_results()

if __name__ == "__main__":
    main()
