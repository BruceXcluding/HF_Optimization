from argparse import ArgumentParser
import os

parser = ArgumentParser()

parser.add_argument("--mode", default="pt_native", required=True, type=str, choices=["pt_native", "xformers"], help="QKV mode")
