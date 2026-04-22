import argparse
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os

import numpy as np
import tensorflow as tf

from configs.params import nyc_params, chicago_params
from lib.utils import get_neigh_index, prepare_data
from model import MYPLAN


def _load_params(dataset: str):
    dataset = str(dataset).lower()
    if dataset == "nyc":
        return nyc_params
    if dataset == "chicago":
        return chicago_params
    raise ValueError(f"Unknown dataset: {dataset}")


def _safe_load_dict_xy(path: str):
    try:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            # often a pickled dict stored as 0-d array
            obj = obj.item()
        if isinstance(obj, dict):
            # Normalize to: region_id -> (x, y)
            # Some variants might store: (x, y) -> region_id
            if len(obj) > 0:
                k0 = next(iter(obj.keys()))
                v0 = obj[k0]
                # (x,y)->rid
                if isinstance(k0, (tuple, list, np.ndarray)) and not isinstance(v0, (tuple, list, np.ndarray)):
                    try:
                        inv = {int(v): (int(k[0]), int(k[1])) for k, v in obj.items()}
                        return inv
                    except Exception:
                        pass
            return obj

        # Some datasets store mapping as an ndarray, e.g. shape (R, 2) where each row is (x, y)
        if isinstance(obj, np.ndarray):
            arr = obj
            # If it's an object array that contains a dict
            if arr.dtype == object and arr.size == 1:
                try:
                    maybe = arr.item()
                    if isinstance(maybe, dict):
                        return maybe
                except Exception:
                    pass

            if arr.ndim == 2:
                # (R,2)
                if arr.shape[1] == 2:
                    mapping = {int(i): (int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])}
                    return mapping
                # (2,R)
                if arr.shape[0] == 2:
                    mapping = {int(i): (int(arr[0, i]), int(arr[1, i])) for i in range(arr.shape[1])}
                    return mapping
    except Exception:
        return None
    return None


def _to_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _load_trained_threshold(results_file: str, dataset: str, model: str = None):
    try:
        if not results_file or not os.path.exists(results_file):
            return None
        dataset = str(dataset)
        model = str(model).lower() if model is not None else None

        best_row = None
        best_ts = None
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get('dataset')) != dataset:
                    continue
                if model is not None and str(row.get('model', '')).lower() != model:
                    continue
                ts = row.get('timestamp')
                if best_ts is None or (ts is not None and float(ts) > float(best_ts)):
                    best_ts = float(ts) if ts is not None else best_ts
                    best_row = row
        if not best_row:
            return None
        th = best_row.get('threshold_selected')
        if th is None:
            th = best_row.get('threshold_f1')
        if th is None:
            return None
        return float(th)
    except Exception:
        return None


class TrafficViewerApp:
    def __init__(self, root, initial_dataset="nyc", initial_weights=None, max_neigh=4):
        self.root = root
        self.root.title("TrafficCompare Viewer")

        self.dataset_var = tk.StringVar(value=initial_dataset)
        self.weights_var = tk.StringVar(value=initial_weights or "")
        self.max_neigh_var = tk.IntVar(value=int(max_neigh))
        self.results_file = "results/metrics.jsonl"

        self.status_var = tk.StringVar(value="Idle")

        self.params = None
        self.number_region = None
        self.grid = None
        self.len_recent_time = None
        self.dr = None
        self.number_sp = None

        self.neigh_poi_index = None
        self.neigh_road_index = None
        self.neigh_record_index = None

        self.all_data = None
        self.threshold_nc = None
        self.label = None
        self.dict_xy = None

        self.model = None
        self.pred = None  # [num_windows, R], NaN = not computed

        # For progressive inference (fast when moving forward in time)
        self._infer_last_t = -1
        self._infer_y_dynamic = None

        self.selected_region = tk.IntVar(value=0)
        self.time_index_var = tk.IntVar(value=0)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.view_mode_var = tk.StringVar(value="prob")

        self._build_ui()
        self._on_dataset_change()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        # Dataset
        ttk.Label(frm, text="Dataset").grid(row=0, column=0, sticky="w")
        ds = ttk.OptionMenu(frm, self.dataset_var, self.dataset_var.get(), "nyc", "chicago", command=lambda _: self._on_dataset_change())
        ds.grid(row=0, column=1, sticky="ew")

        # max_neigh
        ttk.Label(frm, text="max_neigh").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frm, from_=1, to=32, textvariable=self.max_neigh_var, width=8).grid(row=1, column=1, sticky="w")

        # weights
        ttk.Label(frm, text="Weights (optional)").grid(row=2, column=0, sticky="w")
        wfrm = ttk.Frame(frm)
        wfrm.grid(row=2, column=1, sticky="ew")
        wfrm.columnconfigure(0, weight=1)
        ttk.Entry(wfrm, textvariable=self.weights_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(wfrm, text="Browse", command=self._browse_weights).grid(row=0, column=1, padx=(6, 0))

        # Actions
        act = ttk.Frame(frm)
        act.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(act, text="Load Data + Build Model", command=self._load_data_and_build).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(act, text="Predict Current", command=self._predict_current_async).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(act, text="Compute All", command=self._compute_all_async).grid(row=0, column=2, padx=(0, 6))

        ttk.Label(act, textvariable=self.status_var).grid(row=0, column=3, sticky="w")

        # Time slider
        ttk.Label(frm, text="Time Index (window)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.time_scale = ttk.Scale(frm, from_=0, to=0, orient="horizontal", command=self._on_time_scale)
        self.time_scale.grid(row=4, column=1, sticky="ew", pady=(10, 0))

        # Threshold + view mode
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="Threshold").grid(row=0, column=0, sticky="w")
        self.th_scale = ttk.Scale(ctrl, from_=0.0, to=1.0, orient="horizontal", command=self._on_threshold_scale)
        self.th_scale.grid(row=0, column=1, sticky="ew")
        self.th_scale.set(float(self.threshold_var.get()))
        ttk.Label(ctrl, textvariable=self.threshold_var, width=6).grid(row=0, column=2, sticky="w", padx=(6, 0))

        ttk.Label(ctrl, text="View").grid(row=0, column=3, sticky="w", padx=(16, 0))
        ttk.OptionMenu(ctrl, self.view_mode_var, self.view_mode_var.get(), "prob", "label", "error", command=lambda _: self._draw_grid()).grid(row=0, column=4, sticky="w")

        # Region selector + output
        mid = ttk.Frame(frm)
        mid.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        mid.columnconfigure(1, weight=1)

        # region list
        left = ttk.Frame(mid)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Label(left, text="Region").grid(row=0, column=0, sticky="w")
        self.region_list = tk.Listbox(left, height=12, exportselection=False)
        self.region_list.grid(row=1, column=0, sticky="ns")
        self.region_list.bind("<<ListboxSelect>>", lambda e: self._on_region_select())

        # display
        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.columnconfigure(0, weight=1)

        self.value_label = ttk.Label(right, text="(not computed)", font=("Segoe UI", 12))
        self.value_label.grid(row=0, column=0, sticky="w")

        self.canvas = tk.Canvas(right, width=520, height=520, bg="white")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

    def _browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select weights file",
            filetypes=[
                ("TensorFlow weights", "*.h5"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.weights_var.set(path)

    def _on_dataset_change(self):
        self.params = _load_params(self.dataset_var.get())
        self.number_region = int(self.params.number_region)
        self.grid = int(getattr(self.params, "grid", 0) or 0)
        self.len_recent_time = int(self.params.len_recent_time)
        self.dr = int(self.params.dr)
        self.number_sp = int(self.params.number_sp)

        self.region_list.delete(0, tk.END)
        for i in range(self.number_region):
            self.region_list.insert(tk.END, str(i))
        if self.number_region > 0:
            self.region_list.selection_set(0)
            self.selected_region.set(0)

        self.pred = None
        self._infer_last_t = -1
        self._infer_y_dynamic = None
        self._update_value_label()
        self._draw_grid()

    def _set_status(self, msg: str):
        # tkinter is not thread-safe; ensure UI updates happen on the main thread
        if threading.current_thread() is threading.main_thread():
            self.status_var.set(msg)
            self.root.update_idletasks()
        else:
            self.root.after(0, lambda: self._set_status(msg))

    def _load_data_and_build(self):
        try:
            self._set_status("Loading data...")
            dataset = self.dataset_var.get()

            all_data_path = f"{dataset}/{self.params.all_data}"
            threshold_path = f"{dataset}/{self.params.threshold_nc}"
            label_path = f"{dataset}/{self.params.label}"
            dict_xy_path = f"{dataset}/{self.params.dict_xy}"

            all_data = np.load(all_data_path)
            threshold_nc = np.load(threshold_path)
            label = np.load(label_path)

            all_data = prepare_data(all_data, self.len_recent_time)
            threshold_nc = prepare_data(threshold_nc, self.len_recent_time)

            label = _to_numpy(label)
            label = np.asarray(label)
            if label.ndim >= 1 and label.shape[0] == int(self.params.len_recent_time) + int(all_data.shape[0]):
                label = label[int(self.params.len_recent_time):]
            if label.ndim > 1 and label.shape[-1] == 1:
                label = np.squeeze(label, axis=-1)

            all_data = _to_numpy(all_data)
            threshold_nc = _to_numpy(threshold_nc)
            self.all_data = np.asarray(all_data, dtype=np.float32)
            self.threshold_nc = np.asarray(threshold_nc, dtype=np.float32)
            self.label = np.asarray(label, dtype=np.float32)
            self.dict_xy = _safe_load_dict_xy(dict_xy_path)

            self._set_status("Building model...")
            max_neigh = int(self.max_neigh_var.get())
            self.neigh_road_index = get_neigh_index(f"{dataset}/road_ad.txt", max_neigh=max_neigh)
            self.neigh_record_index = get_neigh_index(f"{dataset}/record_ad.txt", max_neigh=max_neigh)
            self.neigh_poi_index = get_neigh_index(f"{dataset}/poi_ad.txt", max_neigh=max_neigh)

            self.model = MYPLAN(
                self.dr,
                self.len_recent_time,
                self.number_sp,
                self.number_region,
                self.neigh_poi_index,
                self.neigh_road_index,
                self.neigh_record_index,
                attention_mode="scaled_dot",
                evolution_smooth=False,
            )

            # build weights by running one forward pass
            y_dynamic = tf.ones((self.len_recent_time, self.number_region, 2 * self.dr), dtype=tf.float32)
            _ = self.model(
                tf.convert_to_tensor(self.all_data[:1]),
                tf.convert_to_tensor(self.threshold_nc[:1]),
                y_dynamic,
            )

            weights = str(self.weights_var.get()).strip()
            if weights:
                self._set_status("Loading weights...")
                self.model.load_weights(weights)

            trained_th = _load_trained_threshold(self.results_file, dataset, model="myplan")
            if trained_th is not None:
                self.threshold_var.set(float(trained_th))
                if hasattr(self, 'th_scale'):
                    self.th_scale.set(float(trained_th))

            # Initialize prediction cache: NaN means not computed yet
            num_windows = int(self.all_data.shape[0])
            self.pred = np.full((num_windows, int(self.number_region)), np.nan, dtype=np.float32)
            self._infer_last_t = -1
            self._infer_y_dynamic = tf.ones(
                (self.len_recent_time, int(self.number_region), 2 * self.dr),
                dtype=tf.float32,
            )

            # update time slider range
            max_t = max(0, int(self.all_data.shape[0]) - 1)
            self.time_scale.configure(from_=0, to=max_t)
            self.time_scale.set(0)
            self.time_index_var.set(0)

            self._set_status("Ready")
            if not weights:
                messagebox.showinfo(
                    "Info",
                    "Weights not provided. The model is randomly initialized, so probabilities are not meaningful. "
                    "Provide a trained weights file via 'Browse' for real predictions.",
                )
        except Exception as e:
            self._set_status("Error")
            messagebox.showerror("Error", str(e))

    def _predict_current_async(self):
        if self.model is None or self.all_data is None or self.threshold_nc is None or self.pred is None:
            messagebox.showwarning("Warning", "Please click 'Load Data + Build Model' first.")
            return

        t = int(self.time_index_var.get())
        th = threading.Thread(target=self._predict_time, args=(t,), daemon=True)
        th.start()

    def _compute_all_async(self):
        if self.model is None or self.all_data is None or self.threshold_nc is None or self.pred is None:
            messagebox.showwarning("Warning", "Please click 'Load Data + Build Model' first.")
            return

        th = threading.Thread(target=self._compute_all, daemon=True)
        th.start()

    def _reset_infer_state(self):
        self._infer_last_t = -1
        self._infer_y_dynamic = tf.ones(
            (self.len_recent_time, int(self.number_region), 2 * self.dr),
            dtype=tf.float32,
        )

    def _predict_time(self, target_t: int):
        try:
            num_windows = int(self.all_data.shape[0])
            if target_t < 0 or target_t >= num_windows:
                self._set_status("Error")
                self.root.after(0, lambda: messagebox.showerror("Error", f"time index out of range: {target_t}"))
                return

            # If already computed, just refresh UI
            if not np.isnan(self.pred[target_t]).all():
                self._set_status("Ready")
                self.root.after(0, self._update_value_label)
                self.root.after(0, self._draw_grid)
                return

            # If user jumps backward, reset and recompute from start to target_t
            if target_t <= self._infer_last_t:
                self._reset_infer_state()

            self._set_status(f"Predicting time {target_t}...")

            y_dynamic = self._infer_y_dynamic
            start_t = self._infer_last_t + 1
            for i in range(start_t, target_t + 1):
                x = tf.convert_to_tensor(self.all_data[i : i + 1])
                th_nc = tf.convert_to_tensor(self.threshold_nc[i : i + 1])
                y_pred, y_dynamic_now, _ = self.model(x, th_nc, y_dynamic)

                self.pred[i] = y_pred.numpy().reshape((-1,))
                y_dynamic = y_dynamic_now
                self._infer_last_t = i
                self._infer_y_dynamic = y_dynamic

            self._set_status("Ready")
            self.root.after(0, self._update_value_label)
            self.root.after(0, self._draw_grid)
        except Exception as e:
            self._set_status("Error")
            msg = str(e)
            self.root.after(0, lambda m=msg: messagebox.showerror("Error", m))

    def _compute_all(self):
        try:
            num_windows = int(self.all_data.shape[0])

            # reset inference state and compute from t=0
            self._reset_infer_state()
            y_dynamic = self._infer_y_dynamic

            self._set_status(f"Computing all... 0/{num_windows}")

            for i in range(num_windows):
                x = tf.convert_to_tensor(self.all_data[i : i + 1])
                th_nc = tf.convert_to_tensor(self.threshold_nc[i : i + 1])
                y_pred, y_dynamic_now, _ = self.model(x, th_nc, y_dynamic)

                self.pred[i] = y_pred.numpy().reshape((-1,))
                y_dynamic = y_dynamic_now
                self._infer_last_t = i
                self._infer_y_dynamic = y_dynamic

                if i % 50 == 0:
                    self._set_status(f"Computing all... {i}/{num_windows}")

            self._set_status("All predictions ready")
            self.root.after(0, self._update_value_label)
            self.root.after(0, self._draw_grid)
        except Exception as e:
            self._set_status("Error")
            msg = str(e)
            self.root.after(0, lambda m=msg: messagebox.showerror("Error", m))

    def _on_time_scale(self, val):
        try:
            self.time_index_var.set(int(float(val)))
        except Exception:
            return
        self._update_value_label()
        self._draw_grid()

    def _on_threshold_scale(self, val):
        try:
            self.threshold_var.set(round(float(val), 3))
        except Exception:
            return
        if hasattr(self, 'value_label'):
            self._update_value_label()
            self._draw_grid()

    def _on_region_select(self):
        sel = self.region_list.curselection()
        if not sel:
            return
        self.selected_region.set(int(sel[0]))
        self._update_value_label()

    def _update_value_label(self):
        r = int(self.selected_region.get())
        t = int(self.time_index_var.get())
        if self.pred is None:
            self.value_label.configure(text=f"Prob: (not computed) | Label: (n/a) | time={t} region={r}")
            return
        if t < 0 or t >= self.pred.shape[0]:
            self.value_label.configure(text=f"Prob: (out of range) | Label: (n/a) | time={t} region={r}")
            return

        p_nan = bool(np.isnan(self.pred[t, r]))
        p = None if p_nan else float(self.pred[t, r])

        y = None
        if self.label is not None:
            try:
                if self.label.ndim == 1:
                    y = float(self.label[t])
                elif self.label.ndim == 2:
                    y = float(self.label[t, r])
            except Exception:
                y = None

        th = float(self.threshold_var.get())
        if p is None:
            prob_txt = "(not computed)"
        else:
            prob_txt = f"{p:.6f}"

        if y is None:
            label_txt = "(n/a)"
        else:
            label_txt = f"{y:.0f}" if abs(y - round(y)) < 1e-6 else f"{y:.3f}"

        if p is not None and y is not None:
            pred_cls = int(p > th)
            true_cls = int(y > 0.5)
            ok = (pred_cls == true_cls)
            ok_txt = "OK" if ok else "WRONG"
            self.value_label.configure(text=f"Prob: {prob_txt} | Label: {label_txt} | Pred@{th:.2f}: {pred_cls} | True: {true_cls} | {ok_txt} | time={t} region={r}")
        else:
            self.value_label.configure(text=f"Prob: {prob_txt} | Label: {label_txt} | time={t} region={r}")

    def _draw_grid(self):
        self.canvas.delete("all")
        if self.number_region is None or self.number_region <= 0:
            return

        # If dict_xy is available and looks like {region_id: (x,y)}, draw heatmap on grid
        if isinstance(self.dict_xy, dict) and self.grid:
            t = int(self.time_index_var.get())
            probs = None
            if self.pred is not None and 0 <= t < self.pred.shape[0] and not np.isnan(self.pred[t]).all():
                probs = self.pred[t]

            labels = None
            if self.label is not None:
                try:
                    if self.label.ndim == 2 and 0 <= t < self.label.shape[0]:
                        labels = self.label[t]
                except Exception:
                    labels = None

            mode = str(self.view_mode_var.get()).lower()

            pad = 10
            size = 500
            g = int(self.grid)
            cell = size / float(g)

            self.canvas.create_rectangle(pad, pad, pad + size, pad + size, outline="#cccccc")

            for rid in range(int(self.number_region)):
                xy = self.dict_xy.get(rid) or self.dict_xy.get(str(rid))
                if xy is None:
                    continue
                try:
                    x, y = int(xy[0]), int(xy[1])
                except Exception:
                    continue

                if mode == "label" and labels is not None:
                    v = float(labels[rid])
                elif mode == "error" and (probs is not None) and (labels is not None):
                    v = float(abs(float(probs[rid]) - float(labels[rid])))
                else:
                    v = float(probs[rid]) if probs is not None else 0.0

                v = max(0.0, min(1.0, v))
                red = int(255 * v)
                blue = int(255 * (1.0 - v))
                color = f"#{red:02x}00{blue:02x}"

                x0 = pad + x * cell
                y0 = pad + y * cell
                x1 = x0 + cell
                y1 = y0 + cell
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

            # highlight selected region
            rid = int(self.selected_region.get())
            xy = self.dict_xy.get(rid) or self.dict_xy.get(str(rid))
            if xy is not None:
                try:
                    x, y = int(xy[0]), int(xy[1])
                    x0 = pad + x * cell
                    y0 = pad + y * cell
                    x1 = x0 + cell
                    y1 = y0 + cell
                    outline = "#00aa00"
                    if probs is not None and labels is not None:
                        try:
                            th = float(self.threshold_var.get())
                            pred_cls = int(float(probs[rid]) > th)
                            true_cls = int(float(labels[rid]) > 0.5)
                            outline = "#00aa00" if pred_cls == true_cls else "#cc0000"
                        except Exception:
                            pass
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=2)
                except Exception:
                    pass

            return

        # fallback: no dict_xy, just show info
        self.canvas.create_text(10, 10, anchor="nw", text="dict_xy not available or grid unknown. Use region list.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nyc", choices=["nyc", "chicago"])
    parser.add_argument("--weights", type=str, default="", help="Path to trained weights file (optional)")
    parser.add_argument("--max_neigh", type=int, default=4)
    args = parser.parse_args()

    root = tk.Tk()
    app = TrafficViewerApp(root, initial_dataset=args.dataset, initial_weights=args.weights, max_neigh=args.max_neigh)
    root.mainloop()


if __name__ == "__main__":
    main()
