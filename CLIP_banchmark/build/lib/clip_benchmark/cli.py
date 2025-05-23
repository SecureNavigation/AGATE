"""Console script for clip_benchmark."""
import argparse
import sys
import random
import json
import torch
import csv
from copy import copy
import os
from itertools import product
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import image_caption_selection, zeroshot_classification, zeroshot_retrieval, linear_probe, captioning
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES

def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+", help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="/root/autodl-tmp", type=str, help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"], help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, nargs="+", default=["laion400m_e32"], help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+", help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto", choices=["zeroshot_classification", "zeroshot_retrieval", "linear_probe", "captioning", "image_caption_selection", "auto"], help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True, help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int, help="for retrieval, select the k for Recall@K metric. ", nargs="+",)
    parser_eval.add_argument('--fewshot_k', default=-1, type=int, help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float, help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true", help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--model_cache_dir', default=None, type=str, help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str, help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str, help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str, help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")

    parser_eval.add_argument('--language', default="en", type=str, nargs="+", help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output', default="result.json", type=str, help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str, help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str, help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true", help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str, help="optional cache directory for webdataset only")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str,  nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str,  default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    # watermark
    parser_eval.add_argument('--watermark_dir', default="./watermark/", type=str,
                             help="watermark matrix dir")
    parser_eval.add_argument('--watermark_dim', default=768, type=int,
                             help="the the dimension of watermark matrix")
    parser_eval.add_argument('--trigger_num', default=1024, type=int,
                             help="The number of trigger number")

    args = parser.parse_args()
    return parser, args

def main():
    parser, base = get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)

def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()
    for path in base.files:
        data = json.load(open(path))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        row['model_fullname'] = row['model'] + ' ' + row['pretrained']
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)
    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main_eval(base):
    # Get list of pre-trained models to evaluate
    pretrained_model = _as_list(base.pretrained_model)
    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                # if path, read file, each line is a pre-trained model
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                # if part of `model_collection`, retrieve from it
                models.extend(model_collection[name])
            else:
                # if not, assume it is in the form of `model,pretrained`
                model, pretrained = name.split(',')
                models.append((model, pretrained))
    else:
        models = list(product(base.model, base.pretrained))

    # Ge list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)
    
    # Get list of languages to evaluate on
    languages = _as_list(base.language)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Languages: {languages}")
    runs = product(models, datasets, languages)
    if base.distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        # randomize runs so that runs are balanced across gpus
        random.seed(base.seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]
    for (model, pretrained), (dataset), (language) in runs:
        # We iterative over all possible model/dataset/languages
        args = copy(base)
        args.model = model
        args.pretrained = pretrained
        args.dataset = dataset
        args.language = language
        run(args)

def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l

def run(args):
    """Console script for clip_benchmark."""
    
    if torch.cuda.is_available():
        if args.distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = 'cuda:%d' % local_rank
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        args.device = device
    else:
        args.device = "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = os.path.basename(args.pretrained) if os.path.isfile(args.pretrained) else args.pretrained
    pretrained_slug_full_path = args.pretrained.replace('/', '_') if os.path.isfile(args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        model=args.model, 
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task, 
        dataset=dataset_slug,
        language=args.language
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(f"Running '{task}' on '{dataset_name}' with the model '{args.pretrained}' on language '{args.language}'")
    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
        model.eval()
        dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform, 
            split=args.split, 
            annotation_file=args.annotation_file,
            download=True,
            language=args.language,
            task=task,
            custom_template_file=args.custom_template_file,
            custom_classname_file=args.custom_classname_file,
            wds_cache_dir=args.wds_cache_dir,
        )
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if hasattr(dataset, "classes") and dataset.classes:
                try:
                    print(f"Dataset classes: {dataset.classes}")
                    print(f"Dataset number of classes: {len(dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        if args.dataset.startswith("wds/"):
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(args.batch_size), batch_size=None, 
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, 
                shuffle=False, num_workers=args.num_workers, 
                collate_fn=collate_fn
            )
    if task == "zeroshot_classification":
        zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics = zeroshot_classification.evaluate(
            model, 
            dataloader, 
            tokenizer, 
            classnames, zeroshot_templates,
            watermark_dir=args.watermark_dir,
            watermark_dim=args.watermark_dim,
            trigger_num=args.trigger_num,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            save_clf=args.save_clf,
            load_clfs=args.load_clfs,
        )
    elif task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model, 
            dataloader, 
            tokenizer,
            watermark_dir=args.watermark_dir,
            watermark_dim=args.watermark_dim,
            trigger_num=args.trigger_num,
            recall_k_list=args.recall_k,
            device=args.device, 
            amp=args.amp
        )
    elif task == "image_caption_selection":
        metrics = image_caption_selection.evaluate(
            model,
            dataloader,
            tokenizer,
            device=args.device,
            amp=args.amp,
        )
    elif task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset, 
            root=dataset_root, 
            transform=transform, 
            split='train', 
            annotation_file=args.annotation_file,
            download=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, 
            collate_fn=collate_fn, pin_memory=True,
        )
        metrics = linear_probe.evaluate(
            model,
            train_dataloader, 
            dataloader, 
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
        )
    elif task == "captioning":
        metrics = captioning.evaluate(
            model=model, 
            dataloader=dataloader, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
            transform=transform
        )
    else:
        raise ValueError("Unsupported task: {}. task should be `zeroshot_classification`, `zeroshot_retrieval`, `linear_probe`,  or `captioning`".format(task))
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
