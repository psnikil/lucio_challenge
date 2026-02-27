import pymupdf
import os
import time
import json
from pathlib import Path
import queue
import multiprocessing as mp
import collections
import asyncio
import zlib
import orjson
from llama_cpp import Llama
import faiss
import numpy as np
from typing import List, Sequence, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import torch
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
ROOT = Path("./data/Testing_Set")
EMBEDDER_MODEL_PATH=Path("./models/embedders")
NUM_WORKERS = mp.cpu_count() or 8
DOC_BATCH = 1
QUEUE_MAX = 65536
USE_COMPRESSION = False
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNKS_PER_MESSAGE = 4096
PAGE_RANGE = 25
BATCH_TEXT_SIZE = 10000
# Metrics
ENABLE_FINE_METRICS = True                 # set True if you really need 

@dataclass
class PageRangeJob:
    path: str
    start_page: int
    end_page: int  # exclusive

""" -----------Timer functions------------ """

def now():
    return time.perf_counter()

def report(metrics_q, name, dt, n=1,worker_id=None):
    metrics_q.put((name, dt, n,worker_id))

def collect_metrics(metrics_q, done_event):
    # this calculates the whole time,
    totals = collections.defaultdict(list)
    counts = collections.defaultdict(list)

    while not done_event.is_set() or not metrics_q.empty():
        try:
            name, dt, n, worker_id = metrics_q.get(timeout=0.5)
            totals[name].append((dt, worker_id))
            counts[name].append(n)
        except Exception:
            pass

    print("\n=== PIPELINE TIMING REPORT ===")
    # print('the raw totals are',totals)
    # print('the raw counts are',counts)
    for k in sorted(totals):
        # TODO: fix this
        # avg = sum(totals[k]) / max(1, sum(counts[k]))
        # avg_worker_time = sum(totals[k][]) / max(1, len(totals[k]))
        # print(f"{k:25s}  total={sum(totals[k][0]):8.2f}s  avg={avg:8.4f}s  n={sum(counts[k])}  avg_worker_time={avg_worker_time:8.4f}s")
        # get worker id with max time
        max_worker_time = max(totals[k], key=lambda x: x[0])
        print(f"{k:25s}  max_worker_time={max_worker_time[0]:8.4f}s  worker_id={max_worker_time[1]}")


""" -----------Text Chunking------------ """
# TODO: needs to yield an object you can embed
def chunk_text(text, size=800, overlap=120):
    i = 0
    n = len(text)
    while i < n:
        yield text[i:i+size]
        i += size - overlap

def serialize_chunks(chunks: Sequence[str]) -> bytes:
    """
    Serialise a sequence of chunks for IPC.

    This is factored out so you can easily switch between:
      - plain pickled Python lists (no compression)
      - orjson + optional zlib
    """
    if USE_COMPRESSION:
        return zlib.compress(orjson.dumps(chunks))
    else:
        # plain JSON bytes; mp.Queue will still pickle this bytes object
        return orjson.dumps(chunks)


def deserialize_chunks(payload: bytes) -> List[str]:
    if USE_COMPRESSION:
        return orjson.loads(zlib.decompress(payload))
    else:
        return orjson.loads(payload)



def parse_worker(worker_id, in_q, out_q, metrics_q):
    # os.sched_setaffinity(0, {worker_id})

    while True:
        jobs = in_q.get()
        if jobs is None:
            break

        batch_chunks = []
        total_chunks = 0
        # t0 = now()

        for job in jobs:
            # t_text = 0.0
            # t_open = now()
            doc = pymupdf.open(job.path)
            batch_text = ""
            for page_idx in range(job.start_page, job.end_page):
                page = doc.load_page(page_idx)
                text = page.get_text("text")
                # if ENABLE_FINE_METRICS:
                #     report(metrics_q, "pdf_open", time.perf_counter() - t_open, worker_id=worker_id)
                # t_text += time.perf_counter() - t_open
                
                # t_chunk = 0.0
                # t_c = now()
                batch_text += text
                if len(batch_text) > BATCH_TEXT_SIZE:
                    for c in chunk_text(batch_text):
                        batch_chunks.append(c)
                        total_chunks += 1

                        # if total_chunks % CHUNKS_PER_MESSAGE == 0:
                            # t_ser = now()
            payload = serialize_chunks(batch_chunks)
            # if ENABLE_FINE_METRICS:
            #     report(metrics_q, "ipc_serialize", now() - t_ser, len(batch_chunks), worker_id=worker_id)
            out_q.put(payload)
            batch_chunks = []
                    # t_chunk += now() - t_c

            doc.close()
            # if ENABLE_FINE_METRICS:
            #     report(metrics_q, "text_extract", t_text, worker_id=worker_id)
            #     report(metrics_q, "chunking", t_chunk, worker_id=worker_id)
        
        if batch_chunks:
            # t_ser = now()
            payload = serialize_chunks(batch_chunks)
            # if ENABLE_FINE_METRICS:
            #     report(metrics_q, "ipc_serialize", now() - t_ser, len(batch_chunks), worker_id=worker_id)
            out_q.put(payload)
            batch_chunks = []

        # if ENABLE_FINE_METRICS:
        #     report(metrics_q, "parse_batch_total", now() - t0, total_chunks, worker_id=worker_id)


async def embed_sink(out_q, metrics_q, llm:None,dim:None,index:None):
    # Doing this to test the pipeline without the embedding model
    loop = asyncio.get_running_loop()
    batched_chunks = []
    count = 0
    TEST_MODE = False    
    if TEST_MODE:
        while True:
            payload = await loop.run_in_executor(None, out_q.get)
            if payload is None:
                break
            # t_deser = now()
            chunks = deserialize_chunks(payload)
            # if ENABLE_FINE_METRICS:
            #     report(metrics_q, "ipc_deserialize", now() - t_deser, len(chunks))
    else:

        loop = asyncio.get_running_loop()
        while True:
            payload = await loop.run_in_executor(None, out_q.get)
            # print(len(payload))
            if payload is None:
                embs = await loop.run_in_executor(None, lambda: llm.encode(batched_chunks, batch_size=256, show_progress_bar=True))
                # vectors = np.array(embs, dtype="float32")
                # report(metrics_q, "embedding", time.perf_counter() - t_embed, len(chunks))
                print(embs.shape)
                t_db = time.perf_counter()
                index.add(embs)
                batched_chunks = []


                break

            t_deser = time.perf_counter()

            chunks = deserialize_chunks(payload)
            # print(len(chunks))

            # if chunks:
            batched_chunks.extend(chunks)
            # print(len(batched_chunks))
            # report(metrics_q, "ipc_deserialize", time.perf_counter() - t_deser, len(chunks))
            if (len(batched_chunks) >= count*512*32):
                t_embed = time.perf_counter()
                embs = await loop.run_in_executor(None, lambda: llm.encode(batched_chunks, batch_size=256, show_progress_bar=True))
                # vectors = np.array(embs, dtype="float32")
                # report(metrics_q, "embedding", time.perf_counter() - t_embed, len(chunks))
                print(embs.shape)
                t_db = time.perf_counter()
                index.add(embs)
                batched_chunks = []
                count += 1
            # report(metrics_q, "faiss_add", time.perf_counter() - t_db, len(chunks))

        print("FAISS index size:", index.ntotal)


def sink_process_entry(out_q, metrics_q, llm, dim, index):
    asyncio.run(embed_sink(out_q, metrics_q, llm, dim, index))

def main():
    mp.set_start_method("spawn")

    llm = SentenceTransformer("Daemontatox/all-MiniLM-L6-v2-bnb-4bit", 
                            device="cuda:0",
                            cache_folder=EMBEDDER_MODEL_PATH,
                            trust_remote_code=True,
                            )
    dim = len(llm.encode(["test"])[0])
    print(dim)
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200

    print(f"FAISS index dim={dim}")


    metrics_q = mp.Queue()

    # done_event = mp.Event()
    # collector = mp.Process(target=collect_metrics, args=(metrics_q, done_event))
    # collector.start()

    in_q = mp.Queue(maxsize=QUEUE_MAX)
    out_q = mp.Queue(maxsize=QUEUE_MAX)

    sink_proc = mp.Process(
        target=sink_process_entry,
        args=(out_q, metrics_q, llm, dim, index),
    )
    sink_proc.start()

    workers = [
        mp.Process(target=parse_worker, args=(i, in_q, out_q, metrics_q))
        for i in range(NUM_WORKERS)
    ]

    for w in workers:
        w.start()
    
    start_time = now()

    batch = []
    for root, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith(".pdf"):
                path = os.path.join(root, f)
                doc = pymupdf.open(path)
                n_pages = len(doc)
                doc.close()
                if n_pages <= 25:
                    in_q.put([PageRangeJob(path, 0, n_pages)])
                else:
                    jobs = []
                    for start in range(0, n_pages, PAGE_RANGE):
                        end = min(start + PAGE_RANGE, n_pages)
                        jobs.append(PageRangeJob(path, start, end))
                        in_q.put([PageRangeJob(path, start, end)])
                # if len(batch) == DOC_BATCH:
                #     in_q.put(batch)
                #     batch = []

    if batch:
        in_q.put(batch)

    for _ in workers:
        in_q.put(None)

    for w in workers:
        w.join()

    out_q.put(None)
    sink_proc.join()

    print("Total time: ", time.perf_counter() - start_time)

    # done_event.set()
    # collector.join()



if __name__ == "__main__":
    main()