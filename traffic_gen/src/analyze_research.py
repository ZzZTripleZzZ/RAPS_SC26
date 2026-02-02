import os
import glob
import subprocess
import re
import json
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# ==========================================
# Phase 1: Data Ingestion (Transparent DUMPI Parser)
# ==========================================

class DumpiParser:
    """
    Stateful parser that tracks BOTH supported and unsupported MPI calls.
    Address Comment 2: Explicitly reports coverage bias.
    """
    
    # Supported Operations
    SUPPORTED_OPS = {"MPI_Send", "MPI_Isend", "MPI_Allreduce"}

    # Mapping common MPI types to bytes
    TYPE_SIZE = {
        'MPI_DOUBLE': 8, 'MPI_FLOAT': 4, 
        'MPI_INT': 4, 'MPI_LONG': 8, 'MPI_CHAR': 1, 'MPI_BYTE': 1,
        'MPI_LONG_LONG': 8, 'MPI_LONG_LONG_INT': 8,
        'MPI_UNSIGNED': 4, 'MPI_UNSIGNED_LONG': 8,
        'MPI_SHORT': 2
    }

    def __init__(self, bin_dir):
        self.bin_dir = bin_dir
        self.files = glob.glob(os.path.join(bin_dir, "dumpi-*-*.bin"))
        self.files.sort()
        self.dumpi_bin = "/usr/local/sst-dumpi/bin/dumpi2ascii"
        print(f">>> [Ingestion] Found {len(self.files)} DUMPI binary files.")

    def parse_single_file(self, filepath):
        """
        Returns: (events_list, ignored_counts_dict)
        """
        try:
            fname = os.path.basename(filepath)
            rank_str = fname.split('-')[-1].replace('.bin', '')
            src_rank = int(rank_str)
        except:
            return [], {}

        events = []
        ignored_counts = defaultdict(int) # Track ignored calls per file
        
        cmd = [self.dumpi_bin, filepath]
        
        try:
            # Buffer size optimization
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1024*1024)
            
            current_evt = None 
            
            for line in process.stdout:
                line = line.strip()
                if not line: continue

                # 1. Detect Start of Function
                if " entering at walltime " in line:
                    parts = line.split()
                    func_name = parts[0]
                    
                    # Check if supported
                    if func_name in self.SUPPORTED_OPS:
                        try:
                            time_idx = parts.index("walltime") + 1
                            timestamp = float(parts[time_idx].replace(',', ''))
                            
                            current_evt = {
                                "func": func_name,
                                "time": timestamp,
                                "count": 0,
                                "datatype": "MPI_BYTE",
                                "dest": -1
                            }
                        except:
                            current_evt = None
                    
                    # If NOT supported, but is an MPI call, log it
                    elif func_name.startswith("MPI_"):
                        ignored_counts[func_name] += 1
                        current_evt = None # Ensure we don't parse args for ignored funcs

                # 2. Parse Arguments (Only for supported funcs)
                elif current_evt:
                    if line.startswith(current_evt['func']) and " returning" in line:
                        size = self.TYPE_SIZE.get(current_evt['datatype'], 1)
                        total_bytes = current_evt['count'] * size
                        
                        if current_evt['func'] in ["MPI_Send", "MPI_Isend"]:
                            if current_evt['dest'] != -1:
                                events.append({
                                    "src": src_rank,
                                    "dst": current_evt['dest'],
                                    "bytes": total_bytes,
                                    "time": current_evt['time'],
                                    "type": "P2P"
                                })
                        elif current_evt['func'] == "MPI_Allreduce":
                            events.append({
                                "src": src_rank,
                                "bytes": total_bytes,
                                "time": current_evt['time'],
                                "type": "COLL"
                            })
                        current_evt = None
                    
                    elif "count=" in line:
                        try:
                            current_evt['count'] = int(line.split("count=")[1].split()[0])
                        except: pass

                    elif "dest=" in line:
                        try:
                            current_evt['dest'] = int(line.split("dest=")[1].split()[0])
                        except: pass

                    elif "datatype=" in line:
                        if "(" in line and ")" in line:
                            try:
                                current_evt['datatype'] = line.split('(')[1].split(')')[0]
                            except: pass

            process.wait()
                
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            return [], {}

        return events, ignored_counts

    def parse_all(self):
        all_events = []
        total_ignored = defaultdict(int)
        
        with ProcessPoolExecutor() as executor:
            results = executor.map(self.parse_single_file, self.files)
            for res_events, res_ignored in results:
                all_events.extend(res_events)
                # Merge ignored counts
                for func, count in res_ignored.items():
                    total_ignored[func] += count
        
        print(f">>> [Ingestion] Parsed {len(all_events)} valid events.")
        
        # --- REPORTING FOR COMMENT 2 ---
        print("\n" + "="*40)
        print("MPI COVERAGE REPORT (Transparency Check)")
        print("="*40)
        print(f"Supported Calls: {list(self.SUPPORTED_OPS)}")
        print("-" * 40)
        print("Ignored Calls (Potential Bias Source):")
        if not total_ignored:
            print("  (None - Full Coverage!)")
        else:
            sorted_ignored = sorted(total_ignored.items(), key=lambda x: x[1], reverse=True)
            for func, count in sorted_ignored:
                print(f"  {func:<20}: {count} occurrences")
        print("="*40 + "\n")
        
        return all_events


# ==========================================
# Phase 2: Representation (Traffic Tensor)
# ==========================================

class TrafficTensorBuilder:
    def __init__(self, events, num_ranks, time_bin=0.1):
        self.events = events
        self.num_ranks = num_ranks
        self.time_bin = time_bin

    def build(self):
        if not self.events:
            print("Warning: No events to build tensor.")
            return None, None
            
        min_time = min(e['time'] for e in self.events)
        max_time = max(e['time'] for e in self.events)
        duration = max_time - min_time
        
        if duration < self.time_bin: duration = self.time_bin
            
        num_bins = int(np.ceil(duration / self.time_bin)) + 1
        
        print(f">>> [Representation] Building Tensor: shape=({num_bins}, {self.num_ranks}, {self.num_ranks}), duration={duration:.2f}s")
        
        tensor = np.zeros((num_bins, self.num_ranks, self.num_ranks), dtype=np.float32)
        coll_counter = 0

        for e in self.events:
            t_rel = e['time'] - min_time
            t_idx = int(t_rel / self.time_bin)
            if t_idx >= num_bins: t_idx = num_bins - 1
            
            src = e['src']
            
            if e['type'] == 'P2P':
                dst = e['dst']
                if 0 <= src < self.num_ranks and 0 <= dst < self.num_ranks:
                    tensor[t_idx, src, dst] += e['bytes']
            
            elif e['type'] == 'COLL':
                coll_counter += 1
                avg_bytes = e['bytes'] / self.num_ranks
                tensor[t_idx, src, :] += avg_bytes
                tensor[t_idx, src, src] -= avg_bytes

        print(f">>> [Representation] Processed {coll_counter} collective events.")
        return tensor, num_bins


# ==========================================
# Phase 3: Intelligence (Classification)
# ==========================================

class PatternClassifier:
    def __init__(self, tensor):
        self.tensor = tensor 
        self.num_ranks = tensor.shape[1]

    def extract_static_affinity(self):
        total_matrix = np.sum(self.tensor, axis=0)
        max_val = np.max(total_matrix)
        if max_val == 0: return [], total_matrix

        affinity_matrix = total_matrix / max_val
        edges = []
        for i in range(self.num_ranks):
            for j in range(self.num_ranks):
                if i == j: continue
                # Explicit float cast for JSON
                weight = float(affinity_matrix[i, j])
                if weight > 0.001: 
                    edges.append({"u": i, "v": j, "weight": round(weight, 4)})
        return edges, total_matrix

    def classify(self, total_matrix):
        non_zeros = np.count_nonzero(total_matrix)
        total_elements = self.num_ranks * self.num_ranks
        sparsity = 1.0 - (non_zeros / total_elements)
        
        k = 2 
        diag_traffic = 0
        total_volume = np.sum(total_matrix)
        
        for i in range(self.num_ranks):
            start = max(0, i-k)
            end = min(self.num_ranks, i+k+1)
            diag_traffic += np.sum(total_matrix[i, start:end])
        
        diag_ratio = diag_traffic / total_volume if total_volume > 0 else 0
        
        pattern = "UNKNOWN"
        if sparsity > 0.7:
            if diag_ratio > 0.5: pattern = "STENCIL_NEAREST_NEIGHBOR"
            else: pattern = "SPARSE_RANDOM"
        else:
            pattern = "ALL_TO_ALL_DENSE"

        # Explicit float cast for JSON
        return {
            "pattern_type": pattern,
            "sparsity": float(round(sparsity, 3)),
            "diagonal_dominance": float(round(diag_ratio, 3))
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPS Research Profiler (Transparent Edition)")
    parser.add_argument("--data_dir", type=str, default="../data/lulesh_baseline", help="Directory containing dumpi-*.bin files")
    parser.add_argument("--out_dir", type=str, default="raps_output", help="Directory to save outputs")
    parser.add_argument("--ranks", type=int, default=8, help="Number of MPI ranks")
    parser.add_argument("--bin_size", type=float, default=0.01, help="Time bin size in seconds")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    if "LD_LIBRARY_PATH" not in os.environ:
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/sst-dumpi/lib"
    else:
        os.environ["LD_LIBRARY_PATH"] += ":/usr/local/sst-dumpi/lib"

    print(f"--- Starting Analysis on {args.data_dir} ---")
    dumpi = DumpiParser(args.data_dir)
    if not dumpi.files:
        print("No DUMPI files found.")
        exit(1)
        
    events = dumpi.parse_all()
    
    builder = TrafficTensorBuilder(events, args.ranks, args.bin_size)
    tensor, num_bins = builder.build()
    
    if tensor is not None:
        npy_path = os.path.join(args.out_dir, "dynamic_traffic.npy")
        np.save(npy_path, tensor)
        print(f">>> [Output] Saved Dynamic Tensor to {npy_path}")

        classifier = PatternClassifier(tensor)
        edges, total_matrix = classifier.extract_static_affinity()
        stats = classifier.classify(total_matrix)
        
        json_output = {"job_profile": stats, "affinity_graph": edges}
        
        json_path = os.path.join(args.out_dir, "static_profile.json")
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
            
        print(f">>> [Output] Saved Static Profile to {json_path}")
        print(f"--- Analysis Complete ---")
        print(f"Detected Pattern: {stats['pattern_type']}")
    else:
        print("Analysis failed.")