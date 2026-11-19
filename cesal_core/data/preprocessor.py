import numpy  as np
import pandas as pd
import torch
from tqdm import tqdm

class Preprocessor(object):
    """Turns event logs into fixed-length context windows (adapted from DeepCASE)."""

    def __init__(self, length, timeout, NO_EVENT=-1337):
        """length: context events per window; timeout: max seconds between a context
        event and its event; NO_EVENT: id that pads missing context."""
        self.context_length = length
        self.timeout        = timeout

        self.NO_EVENT = NO_EVENT

        self.REQUIRED_COLUMNS = {'timestamp', 'event', 'machine'}


    def sequence(self, data, labels=None, verbose=False, desc='Loading'):
        """Turn a DataFrame (timestamp, event, machine[, label]) into sequences.

        Returns (context, events, labels, mapping). Events are remapped to
        0..n_events-1 and ``mapping`` translates them back. An int ``labels`` labels
        every row; labels is None if neither ``labels`` nor a 'label' column exists.
        """

        if isinstance(labels, int):
            labels = np.full(data.shape[0], labels, dtype=int)

        labels = np.asarray(labels)

        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                ".csv file must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        if labels.ndim and labels.shape[0] != data.shape[0]:
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )

        mapping = {
            i: event for i, event in enumerate(np.unique(data['event'].values))
        }

        if self.NO_EVENT in mapping.values():
            raise ValueError(
                "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
            )

        mapping[len(mapping)] = self.NO_EVENT
        mapping_inverse = {v: k for k, v in mapping.items()}

        data['event'] = data['event'].map(mapping_inverse)

        events = torch.Tensor(data['event'].values).to(torch.long)

        context = torch.full(
            size       = (data.shape[0], self.context_length),
            fill_value = mapping_inverse[self.NO_EVENT],
        ).to(torch.long)

        if labels.ndim:
            labels = torch.Tensor(labels).to(torch.long)
        elif 'label' in data.columns:
            labels = torch.Tensor(data['label'].values).to(torch.long)
        else:
            labels = None

        data = data.sort_values(by='timestamp')

        machine_grouped = data.groupby('machine')
        if verbose: machine_grouped = tqdm(machine_grouped, desc=desc)

        for machine, events_ in machine_grouped:
            indices    = events_.index.values
            timestamps = events_['timestamp'].values
            events_    = events_['event'].values

            machine_context = np.full(
                (events_.shape[0], self.context_length),
                mapping_inverse[self.NO_EVENT],
                dtype = int,
            )

            for i in range(self.context_length):
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                timeout_mask = time_diff > self.timeout

                machine_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_EVENT],
                    events_[:-i-1],
                )

            machine_context = torch.Tensor(machine_context).to(torch.long)
            context[indices] = machine_context

        return context, events, labels, mapping


    def csv(self, path, nrows=None, labels=None, verbose=False, desc='Loading'):
        """Sequence a CSV with columns timestamp, machine, event[, label]."""
        data = pd.read_csv(path, nrows=nrows)

        return self.sequence(data, labels=labels, verbose=verbose, desc=desc)


    def text(self, path, nrows=None, labels=None, verbose=False, desc='Loading'):
        """Sequence a text file: one machine per line, space-separated event ids."""
        events     = list()
        machines   = list()

        with open(path) as infile:
            for machine, line in enumerate(infile):
                if nrows is not None and machine >= nrows: break

                for event in map(int, line.split()):
                    events  .append(event)
                    machines.append(machine)

        data = pd.DataFrame({
            'timestamp': np.arange(len(events)),
            'event'    : events,
            'machine'  : machines,
        })

        return self.sequence(data, labels=labels, verbose=verbose, desc=desc)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Preprocessor: processes log data into fixed-length sequences.",
    )

    parser.add_argument('file',                                  help='file      to preprocess')
    parser.add_argument('--write',                               help='file      to write output')
    parser.add_argument('--type',              default='auto'  , help="file type to preprocess (auto|csv|json|ndjson|t(e)xt)")
    parser.add_argument('--context', type=int, default=10      , help="size of context")
    parser.add_argument('--timeout', type=int, default=60*60*24, help="maximum time between context and event")

    args = parser.parse_args()

    ALLOWED_EXTENSIONS = {'csv', 'txt', 'text'}

    if args.type == 'auto':
        args.type = os.path.splitext(args.file)[1][1:]
        if args.type not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "Automatically parsed extension not supported: '.{}'. "
                "Please manually specify --type (csv|json|ndjson|t(e)xt)"
                .format(args.type)
            )

    preprocessor = Preprocessor(
        length=args.context,
        timeout=args.timeout,
    )

    if args.type == 'csv':
        events, context, labels = preprocessor.csv(args.file)
    elif args.type == 'txt' or args.type == 'text':
        events, context, labels = preprocessor.text(args.file)
    else:
        raise ValueError("Unsupported file type: '{}'".format(args.type))

    if args.write:

        with open(args.write, 'wb') as outfile:
            torch.save({
                'events' : events,
                'context': context,
                'labels' : labels,
            }, outfile)

        with open(args.write, 'rb') as infile:
            data = torch.load(infile)
            events  = data.get('events')
            context = data.get('context')
            labels  = data.get('labels')

    print("Events : {}".format(events))
    print("Context: {}".format(context))
    print("Labels : {}".format(labels))