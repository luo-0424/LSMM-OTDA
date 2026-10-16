# run.py
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*networkx backend defined more than once.*")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import csv
import gc
import time
import ctypes
import random
import pickle
import argparse
from datetime import datetime, timedelta
from functools import partial
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ot
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "pys"))

from pys.gmm import *
from pys.gmm_otda import *
from pys.model import *
from pys.prob_utils import *

class OTMathError(Exception):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Full Multi-Dimensional Grid Search for LSMM-OTDA")
    parser.add_argument('--base_path', type=str, default="./data", help="Path to feature data files")
    parser.add_argument('--results_path', type=str, default="./results", help="Directory to save output CSV/XLSX files")
    parser.add_argument('--features', type=str, default="vit", choices=["resnet101", "resnet50", "vit"], help="Feature backbone name")
    
    parser.add_argument(
        '--clusters_per_class_list', type=int, nargs='+', default=[4],
        help="List of clusters per class"
    )
    parser.add_argument(
        '--reg_e_list', type=float, nargs='+', default=[0.05],
        help="List of reg_e (0.0 triggers EMD, >0 triggers Sinkhorn)"
    )
    parser.add_argument(
        '--q_values', type=float, nargs='+', default=[2.9],
        help="List of q values"
    )
    parser.add_argument(
        '--lr_list', type=float, nargs='+', default=[1e-5],
        help="List of learning rates"
    )
    parser.add_argument(
        '--batch_size_list', type=int, nargs='+', default=[128],
        help="List of batch sizes"
    )
    parser.add_argument('--n_epochs_perceptron', type=int, default=10, help="Number of epochs for shallow neural net")
    
    parser.add_argument(
        '--n_workers', type=int, default=8,
        help="Parallel CPU workers"
    )
    return parser.parse_args()

def sinkhorn(a, b, C, reg_e):
    is_torch = isinstance(C, torch.Tensor)
    device = C.device if is_torch else None
    dtype = C.dtype if is_torch else None
    
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    C_np = C.detach().cpu().numpy() if isinstance(C, torch.Tensor) else np.asarray(C)
    
    C_max = C_np.max()
    C_norm = C_np / C_max if C_max > 1e-12 else C_np
    
    a_sum = a_np.sum()
    b_sum = b_np.sum()
    a_np = a_np / a_sum if a_sum > 1e-12 else np.ones_like(a_np) / len(a_np)
    b_np = b_np / b_sum if b_sum > 1e-12 else np.ones_like(b_np) / len(b_np)
    
    P = None
    errors = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            P = ot.sinkhorn(
                a_np, b_np, C_norm,
                reg=reg_e,
                method='sinkhorn_log',
                numItermax=5000,
                stopThr=1e-7
            )
            if P is not None and not np.isnan(P).any() and not np.all(P == 0):
                return torch.from_numpy(P).to(device=device, dtype=dtype) if is_torch else P
        except Exception as e:
            errors.append(f"sinkhorn_log failed: {e}")
            
        try:
            P = ot.bregman.sinkhorn_stabilized(
                a_np, b_np, C_norm,
                reg=reg_e,
                numItermax=5000,
                stopThr=1e-7
            )
            if P is not None and not np.isnan(P).any() and not np.all(P == 0):
                return torch.from_numpy(P).to(device=device, dtype=dtype) if is_torch else P
        except Exception as e:
            errors.append(f"sinkhorn_stabilized failed: {e}")
            
        if reg_e < 0.05:
            relaxed_reg = 0.05
            try:
                P = ot.sinkhorn(
                    a_np, b_np, C_norm,
                    reg=relaxed_reg,
                    method='sinkhorn_log',
                    numItermax=3000,
                    stopThr=1e-6
                )
                if P is not None and not np.isnan(P).any() and not np.all(P == 0):
                    return torch.from_numpy(P).to(device=device, dtype=dtype) if is_torch else P
            except Exception as e:
                errors.append(f"Relaxed sinkhorn_log (reg=0.05) failed: {e}")
                
    raise OTMathError(f"All Sinkhorn solvers failed for reg_e={reg_e}. Solver Logs: {errors}")

def append_to_csv_realtime(filepath, record, headers):
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        f.flush()

def load_completed_tasks(filepath):
    completed = set()
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            df_exist = pd.read_csv(filepath)
            required_cols = ["clusters_per_class", "reg_e", "q", "lr", "batch_size"]
            if all(col in df_exist.columns for col in required_cols):
                for _, row in df_exist.iterrows():
                    key = (
                        int(row["clusters_per_class"]),
                        round(float(row["reg_e"]), 4),
                        round(float(row["q"]), 2),
                        float(row["lr"]),
                        int(row["batch_size"])
                    )
                    completed.add(key)
        except Exception as e:
            print(f"[Warning] Failed to read existing checkpoint file: {e}")
    return completed

def save_df_to_excel(df, xlsx_path):
    try:
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
    except Exception as e:
        print(f"[Warning] Failed to export Excel file '{xlsx_path}': {e}")

def run_single_task(task_params, args, Xs_shared, Ys_shared, Xt_shared, Yt_shared, yt_shared, n_classes, n_dim, num_threads_per_worker):
    clusters_per_class, reg_e, q, lr_perceptron, batch_size_perceptron = task_params
    q = round(q, 2)
    reg_e = round(float(reg_e), 4)
    lr_perceptron = float(lr_perceptron)
    batch_size_perceptron = int(batch_size_perceptron)

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings('ignore')

    seed_ = 42
    pl.seed_everything(seed_, workers=True)

    Xs = Xs_shared.clone()
    Ys = Ys_shared
    Xt = Xt_shared
    Yt = Yt_shared
    yt = yt_shared

    try:
        clustering_source = partial(conditional_em_gmm, n_clusters=clusters_per_class, random_state=seed_)
        clustering_target = partial(em_gmm, n_clusters=clusters_per_class * n_classes, random_state=seed_)

        if reg_e == 0:
            ot_solver = ot.emd
        else:
            ot_solver = partial(sinkhorn, reg_e=reg_e)

        gc.collect()

        otda = GMMOTDA(
            clustering_source=clustering_source,
            clustering_target=clustering_target,
            ot_solver=ot_solver,
            min_var=1.0,
            q_=q
        )
        otda.fit(Xs, Ys, Xt, Yt)

        numel = 2 * n_classes * clusters_per_class
        w, TXs, TYs = otda.transport_samples(Xs, Ys, numel=numel)

        if TXs.shape[0] == 0:
            return {
                "features": args.features.lower(), "clusters_per_class": clusters_per_class,
                "reg_e": reg_e, "q": q, "lr": lr_perceptron, "batch_size": batch_size_perceptron,
                "max_epoch": 0, "val_max_acc": 0.0, "val_last_acc": 0.0, "nn_acc": 0.0, "map_acc": 0.0
            }

        train_dataset = torch.utils.data.TensorDataset(w, TXs, TYs)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_perceptron, shuffle=True, num_workers=0
        )

        u = torch.ones(len(Xt), device=Xt.device, dtype=Xt.dtype) / len(Xt)
        test_dataset = torch.utils.data.TensorDataset(u, Xt, Yt)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=0
        )

        model = WeightedShallowNeuralNet(
            n_features=n_dim,
            n_classes=n_classes,
            learning_rate=lr_perceptron,
            l2_penalty=0.0,
            momentum=0.9,
            optimizer_name='sgd',
            log_gradients=False,
            max_norm=None
        )

        trainer = pl.Trainer(
            max_epochs=args.n_epochs_perceptron,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )

        trainer.fit(model, train_dataloader, test_dataloader)

        val_max_acc = max(model.history['val_acc']).item() * 100
        max_epoch = int(np.argmax(model.history['val_acc']))
        val_last_acc = model.history['val_acc'][-1].item() * 100

        with torch.no_grad():
            yp = model(Xt.float()).argmax(dim=1)
            nn_acc = round(100 * accuracy_score(yp.cpu().numpy(), yt.cpu().numpy()), 2)

        yp_map = otda.predict_target_labels(Xt, use_estimated_labels=True).argmax(dim=1)
        map_acc = round(100 * accuracy_score(yp_map.cpu().numpy(), Yt.argmax(dim=1).cpu().numpy()), 2)

        del model, trainer, otda, train_dataset, test_dataset, train_dataloader, test_dataloader
        del Xs, Ys, Xt, Yt, yt
        gc.collect()

        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

        return {
            "features": args.features.lower(),
            "clusters_per_class": clusters_per_class,
            "reg_e": reg_e,
            "q": q,
            "lr": lr_perceptron,
            "batch_size": batch_size_perceptron,
            "max_epoch": max_epoch,
            "val_max_acc": round(val_max_acc, 2),
            "val_last_acc": round(val_last_acc, 2),
            "nn_acc": nn_acc,
            "map_acc": map_acc
        }

    except OTMathError as e:
        return {
            "features": args.features.lower(), "clusters_per_class": clusters_per_class,
            "reg_e": reg_e, "q": q, "lr": lr_perceptron, "batch_size": batch_size_perceptron,
            "max_epoch": 0, "val_max_acc": 0.0, "val_last_acc": 0.0, "nn_acc": 0.0, "map_acc": 0.0
        }
    except RuntimeError as e:
        print(f"\n[Unexpected RuntimeError] Task K={clusters_per_class}, reg={reg_e}, q={q} failed: {e}", flush=True)
        return None
    except Exception as e:
        print(f"\n[Unexpected System Error] Task K={clusters_per_class}, reg={reg_e}, q={q} failed: {e}", flush=True)
        return None

def main():
    args = parse_args()
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.perf_counter()

    if args.n_workers > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    n_classes = 12
    clusters_list = sorted(list(set(args.clusters_per_class_list)))
    reg_e_list = sorted([round(r, 4) for r in set(args.reg_e_list)])
    q_values = [round(q, 2) for q in args.q_values]
    lr_list = sorted(list(set(args.lr_list)))
    batch_size_list = sorted(list(set(args.batch_size_list)))
    feat_name = args.features.lower()

    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    main_csv_file = os.path.join(results_path, f"results_{feat_name}_full.csv")
    main_xlsx_file = os.path.join(results_path, f"results_{feat_name}_full.xlsx")

    detailed_headers = [
        "features", "clusters_per_class", "reg_e", "q", "lr", "batch_size",
        "max_epoch", "val_max_acc", "val_last_acc", "nn_acc", "map_acc"
    ]

    all_task_grid = []
    for c in clusters_list:
        n_train = 2 * n_classes * c
        for reg in reg_e_list:
            for q in q_values:
                for lr in lr_list:
                    effective_bs_list = []
                    for bs in batch_size_list:
                        effective_bs_list.append(bs)
                        if bs >= n_train:
                            break
                    for bs in effective_bs_list:
                        all_task_grid.append((c, reg, q, lr, bs))
    total_grid_size = len(all_task_grid)

    completed_keys = load_completed_tasks(main_csv_file)
    pending_tasks = [
        task for task in all_task_grid
        if (task[0], round(task[1], 4), round(task[2], 2), float(task[3]), int(task[4])) not in completed_keys
    ]

    print("\n=================== EXPERIMENT & RESUME CONFIG ===================")
    print(f" Features Backbone       : {feat_name}")
    print(f" Clusters Range (K)      : {clusters_list}")
    print(f" Regularization reg_e    : {reg_e_list}")
    print(f" q Range                 : {q_values[0]} -> {q_values[-1]} ({len(q_values)} points)")
    print(f" Learning Rates (lr)     : {lr_list}")
    print(f" Batch Sizes             : {batch_size_list}")
    print(f" Total Grid Combinations : {total_grid_size} tasks")
    print(f" Already Completed       : {len(completed_keys)} tasks (Skipped)")
    print(f" Remaining Tasks to Run  : {len(pending_tasks)} tasks")
    print(f" Worker Process Count    : {args.n_workers} Workers (spawn)")
    print(f" Realtime Saved CSV      : {main_csv_file}")
    print("==================================================================\n")

    if len(pending_tasks) == 0:
        print("[Info] All tasks in this grid search have already been completed!")
    else:
        if feat_name == "resnet101":
            n_dim = 2048
            filename = "visda_resnet_101.pkl"
        elif feat_name == "resnet50":
            n_dim = 2048
            filename = "visda_resnet_50.pkl"
        elif feat_name == "vit":
            n_dim = 768
            filename = "visda_vit_16.pkl"
        else:
            raise ValueError(f"Invalid feature name '{feat_name}'")

        data_file = os.path.join(args.base_path, filename)
        if not os.path.exists(data_file):
            print(f"[Error] Data file {data_file} not found!")
            return

        with open(data_file, 'rb') as f:
            dataset = pickle.load(f)

        Xs, ys = dataset['Train']
        Xt, yt = dataset['Test']

        Xs = torch.from_numpy(Xs) if isinstance(Xs, np.ndarray) else Xs
        ys = torch.from_numpy(ys) if isinstance(ys, np.ndarray) else ys
        Xt = torch.from_numpy(Xt) if isinstance(Xt, np.ndarray) else Xt
        yt = torch.from_numpy(yt) if isinstance(yt, np.ndarray) else yt

        mean, std = Xs.mean(), Xs.std()
        Xs = (Xs - mean) / (std + 1e-9)
        Xt = (Xt - mean) / (std + 1e-9)

        Ys = torch.nn.functional.one_hot(ys.long(), num_classes=n_classes).float()
        Yt = torch.nn.functional.one_hot(yt.long(), num_classes=n_classes).float()

        Xs.share_memory_()
        Ys.share_memory_()
        Xt.share_memory_()
        Yt.share_memory_()
        yt.share_memory_()

        num_threads_per_worker = 1

        if args.n_workers <= 1:
            for idx, task in enumerate(pending_tasks, 1):
                res = run_single_task(task, args, Xs, Ys, Xt, Yt, yt, n_classes, n_dim, num_threads_per_worker)
                if res is not None:
                    append_to_csv_realtime(main_csv_file, res, detailed_headers)
                    if res['nn_acc'] == 0.0 and res['map_acc'] == 0.0:
                        print(f"[{idx:04d}/{len(pending_tasks)}] K={res['clusters_per_class']} | reg={res['reg_e']} | q={res['q']:.2f} | lr={res['lr']} | bs={res['batch_size']} | [SOLVER FAILED] Recorded as 0.0", flush=True)
                    else:
                        print(f"[{idx:04d}/{len(pending_tasks)}] K={res['clusters_per_class']} | reg={res['reg_e']} | q={res['q']:.2f} | lr={res['lr']} | bs={res['batch_size']} | NN: {res['nn_acc']:.2f}% | MAP: {res['map_acc']:.2f}%", flush=True)
        else:
            spawn_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=spawn_context) as executor:
                future_to_task = {
                    executor.submit(
                        run_single_task, task, args, Xs, Ys, Xt, Yt, yt, n_classes, n_dim, num_threads_per_worker
                    ): task for task in pending_tasks
                }
                
                for idx, future in enumerate(as_completed(list(future_to_task)), 1):
                    task = future_to_task.pop(future) 
                    try:
                        res = future.result()
                        if res is not None:
                            append_to_csv_realtime(main_csv_file, res, detailed_headers)
                            if res['nn_acc'] == 0.0 and res['map_acc'] == 0.0:
                                print(f"[Done {idx:04d}/{len(pending_tasks)}] K={res['clusters_per_class']} | reg={res['reg_e']} | q={res['q']:.2f} | lr={res['lr']} | bs={res['batch_size']} | [SOLVER FAILED] Recorded as 0.0", flush=True)
                            else:
                                print(f"[Done {idx:04d}/{len(pending_tasks)}] K={res['clusters_per_class']} | reg={res['reg_e']} | q={res['q']:.2f} | lr={res['lr']} | bs={res['batch_size']} | NN: {res['nn_acc']:.2f}% | MAP: {res['map_acc']:.2f}%", flush=True)
                        else:
                            print(f"[Skipped {idx:04d}/{len(pending_tasks)}] Task {task} encountered unknown system exception. Skipped recording.", flush=True)
                    except Exception as exc:
                        print(f"[Error] Task {task} failed to yield result with exception: {exc}", flush=True)
                    
                    if idx % 100 == 0:
                        gc.collect()

    if os.path.exists(main_csv_file):
        df_full = pd.read_csv(main_csv_file).sort_values(
            by=["clusters_per_class", "reg_e", "q", "lr", "batch_size"]
        ).reset_index(drop=True)
        df_full.to_csv(main_csv_file, index=False, encoding='utf-8-sig')
        save_df_to_excel(df_full, main_xlsx_file)

        elapsed = time.perf_counter() - start_time
        formatted_time = str(timedelta(seconds=int(elapsed)))
        print(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"耗时统计: {elapsed / 3600:.2f} 小时 ({int(elapsed)} 秒 / {formatted_time})")

        top5_nn = df_full.sort_values(by="nn_acc", ascending=False).head(5)
        top5_map = (
            df_full.sort_values(by="map_acc", ascending=False)
            .drop_duplicates(subset=["clusters_per_class", "reg_e", "q"], keep='first')
            .head(5)
        )

        print("\n" + "=" * 85)
        print("                  TOP 5 BEST Shallow NN ACCURACY RESULTS                      ")
        print("=" * 85)
        for idx, row in enumerate(top5_nn.itertuples(), 1):
            print(f" Rank #{idx} | NN Acc: {row.nn_acc:.2f}% -> K = {row.clusters_per_class}, reg_e = {row.reg_e}, q = {row.q:.2f}, lr = {row.lr}, bs = {row.batch_size}")

        print("\n" + "=" * 85)
        print("                TOP 5 BEST GMM-OTDA MAP ACCURACY RESULTS                      ")
        print("=" * 85)
        for idx, row in enumerate(top5_map.itertuples(), 1):
            print(f" Rank #{idx} | MAP Acc: {row.map_acc:.2f}% -> K = {row.clusters_per_class}, reg_e = {row.reg_e}, q = {row.q:.2f}")

        print("\n" + "=" * 85)
        print(f" Complete Grid Search Dataset Saved To:")
        print(f"   - CSV  : {main_csv_file}")
        print(f"   - XLSX : {main_xlsx_file}")
        print("=" * 85 + "\n")

if __name__ == "__main__":
    main()