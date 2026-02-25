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

ROOT = Path("./data/Testing_Set")
NUM_WORKERS = 8
DOC_BATCH = 1
QUEUE_MAX = 65536
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

""" -----------Timer functions------------ """

def now():
    return time.perf_counter()

def report(metrics_q, name, dt, n=1):
    metrics_q.put((name, dt, n))

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

def chunk_text(text, size=800, overlap=120):
    i = 0
    n = len(text)
    while i < n:
        yield text[i:i+size]
        i += size - overlap

def report(metrics_q, name, dt, n=1, worker_id=None):
    metrics_q.put((name, dt, n, worker_id))

def parse_worker(worker_id, in_q, out_q, metrics_q):
    # os.sched_setaffinity(0, {worker_id})

    while True:
        paths = in_q.get()
        if paths is None:
            break

        batch_chunks = []
        t0 = time.perf_counter()

        for path in paths:
            t_open = time.perf_counter()
            doc = pymupdf.open(path)
            report(metrics_q, "pdf_open", time.perf_counter() - t_open, worker_id=worker_id)

            t_text = 0.0
            t_chunk = 0.0

            for page in doc:
                t_get = time.perf_counter()
                text = page.get_text("text")
                t_text += time.perf_counter() - t_get

                t_c = time.perf_counter()
                for c in chunk_text(text):
                    batch_chunks.append(c)
                t_chunk += time.perf_counter() - t_c

            doc.close()
            report(metrics_q, "text_extract", t_text, worker_id=worker_id)
            report(metrics_q, "chunking", t_chunk, worker_id=worker_id)

        t_ser = time.perf_counter()
        payload = zlib.compress(orjson.dumps(batch_chunks))
        report(metrics_q, "ipc_serialize", time.perf_counter() - t_ser, len(batch_chunks), worker_id=worker_id)

        out_q.put(payload)

        report(metrics_q, "parse_batch_total", time.perf_counter() - t0, len(batch_chunks), worker_id=worker_id)


async def embed_sink(out_q, metrics_q):
    # Doing this to test the pipeline without the embedding model
    loop = asyncio.get_running_loop()
    if True:
        while True:
            payload = await loop.run_in_executor(None, out_q.get)
            if payload is None:
                break
            t_deser = time.perf_counter()
            chunks = orjson.loads(zlib.decompress(payload))
            report(metrics_q, "ipc_deserialize", time.perf_counter() - t_deser, len(chunks))
    else:
        llm = Llama(
            model_path="./models/embedding.gguf",
            n_threads=8,
            n_batch=2048,
            embedding=True,
            use_mmap=True,
            use_mlock=True,
        )

        loop = asyncio.get_running_loop()

        while True:
            payload = await loop.run_in_executor(None, out_q.get)
            if payload is None:
                break

            t_deser = time.perf_counter()
            chunks = orjson.loads(zlib.decompress(payload))
            report(metrics_q, "ipc_deserialize", time.perf_counter() - t_deser, len(chunks))

            t_embed = time.perf_counter()
            embeddings = await loop.run_in_executor(
                None,
                lambda: llm.create_embedding(chunks)
            )
            report(metrics_q, "embedding", time.perf_counter() - t_embed, len(chunks))

            t_sink = time.perf_counter()
            # TODO: write to RAM vector DB (FAISS / Qdrant / custom mmap store)
            report(metrics_q, "db_sink", time.perf_counter() - t_sink, len(chunks))


# def collect_metrics(metrics_q, done_event):
#     totals = collections.defaultdict(float)
#     counts = collections.defaultdict(int)

#     while not done_event.is_set() or not metrics_q.empty():
#         try:
#             name, dt, n = metrics_q.get(timeout=0.5)
#             totals[name] += dt
#             counts[name] += n
#         except Exception:
#             pass

#     print("\n=== PIPELINE TIMING REPORT ===")
#     for k in sorted(totals):
#         avg = totals[k] / max(1, counts[k])
#         print(f"{k:25s}  total={totals[k]:8.2f}s  avg={avg:8.6f}s  n={counts[k]}")

def sink_process_entry(out_q, metrics_q):
    asyncio.run(embed_sink(out_q, metrics_q))

def main():
    mp.set_start_method("spawn")
    metrics_q = mp.Queue()

    done_event = mp.Event()
    collector = mp.Process(target=collect_metrics, args=(metrics_q, done_event))
    collector.start()

    start_time = time.perf_counter()
    in_q = mp.Queue(maxsize=QUEUE_MAX)
    out_q = mp.Queue(maxsize=QUEUE_MAX)

    sink_proc = mp.Process(
        target=sink_process_entry,
        args=(out_q, metrics_q),
    )
    sink_proc.start()

    workers = [
        mp.Process(target=parse_worker, args=(i, in_q, out_q, metrics_q))
        for i in range(NUM_WORKERS)
    ]

    for w in workers:
        w.start()

    batch = []
    for root, _, files in os.walk(ROOT):
        print('the files are', files)
        for f in files:
            if f.endswith(".pdf"):
                batch.append(os.path.join(root, f))
                if len(batch) == DOC_BATCH:
                    in_q.put(batch)
                    batch = []

    if batch:
        in_q.put(batch)

    for _ in workers:
        in_q.put(None)

    for w in workers:
        w.join()

    out_q.put(None)
    sink_proc.join()

    print("Total time: ", time.perf_counter() - start_time)

    done_event.set()
    collector.join()



if __name__ == "__main__":
    main()