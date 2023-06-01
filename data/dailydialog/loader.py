from datasets import load_dataset
import ndjson
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sep', type=str, default="[EOT]")
parser.add_argument('--n', type=int, default=5000)


def create_dataset(n=5000, sep="[EOT]", seed=42):

    random.seed(seed)

    # Load dataset and sample n examples
    dataset = load_dataset('daily_dialog')
    ddict = dataset['train'].to_dict()
    dialogues = ddict['dialog']
    emotions = ddict['emotion']
    acts = ddict['act']
    sample_idx = random.sample(list(range(len(dialogues))),n)

    # Append to data dictionary
    out = []
    for s in sample_idx:
        sdict = {'id': f'dailydialog-{s}',
                'source': f' {sep} '.join(dialogues[s][:-1]).replace('\n', ''),
                'human_completions': [dialogues[s][-1]],
                'annotations': {'n-turns': len(dialogues[s][:-1]),
                                'source-emo': emotions[s][:-1],
                                'comp-emo': emotions[s][-1],
                                'source-act': acts[s][:-1],
                                'comp-act': acts[s][-1]}}
        out.append(sdict)

    # Save as ndjson
    with open('data.ndjson', 'w') as f:
        ndjson.dump(out, f)


if __name__=='__main__':
    args = parser.parse_args()
    create_dataset(n=args.n, sep=args.sep)