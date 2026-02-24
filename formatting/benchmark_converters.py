import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import docling.document_converter as ddc
import pymupdf4llm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from kreuzberg import extract_file
import asyncio
from rapidocr_onnxruntime import RapidOCR

# Configuration
TEST_DATA_DIR = Path("./data/Testing Set")
OUTPUT_DIR = Path("./formatting/benchmark_results")
EXCLUDED_EXTENSIONS = [".xlsx"]

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class Benchmark:
    def __init__(self):
        self.results = {}
        print("Loading Docling models...")
        self.docling_converter = ddc.DocumentConverter()
        print("Loading Marker models...")
        self.marker_model_dict = create_model_dict()
        self.marker_converter = PdfConverter(self.marker_model_dict)
        print("Loading RapidOCR models...")
        self.rapid_ocr = RapidOCR()
        print("Models loaded successfully.")

    def benchmark_docling(self, file_path: Path):
        start_time = time.time()
        try:
            result = self.docling_converter.convert(str(file_path))
            # Just to ensure we've processed it
            _ = result.document.export_to_markdown()
            status = "success"
        except Exception as e:
            status = f"error: {str(e)}"
        
        return time.time() - start_time, status

    def benchmark_pymupdf4llm(self, file_path: Path):
        if file_path.suffix.lower() != ".pdf":
            return 0, "skipped (not pdf)"
        
        start_time = time.time()
        try:
            _ = pymupdf4llm.to_markdown(str(file_path))
            status = "success"
        except Exception as e:
            status = f"error: {str(e)}"
        
        return time.time() - start_time, status

    def benchmark_marker(self, file_path: Path):
        if file_path.suffix.lower() != ".pdf":
            return 0, "skipped (not pdf)"
        
        start_time = time.time()
        try:
            rendered = self.marker_converter(str(file_path))
            _ = rendered.markdown
            status = "success"
        except Exception as e:
            status = f"error: {str(e)}"
        
        return time.time() - start_time, status

    async def benchmark_kreuzberg(self, file_path: Path):
        start_time = time.time()
        try:
            _ = await extract_file(file_path)
            status = "success"
        except Exception as e:
            status = f"error: {str(e)}"
        
        return time.time() - start_time, status

    def benchmark_rapidocr(self, file_path: Path):
        # RapidOCR usually takes images, but we can use it on PDFs by rendering or use its wrapper if available.
        # Here we just measure its raw performance on the first page if it's a PDF (as a proxy for OCR speed)
        if file_path.suffix.lower() != ".pdf":
            return 0, "skipped (not pdf)"
        
        import fitz # PyMuPDF
        start_time = time.time()
        try:
            doc = fitz.open(str(file_path))
            page = doc.load_page(0)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            result, _ = self.rapid_ocr(img_bytes)
            status = "success"
        except Exception as e:
            status = f"error: {str(e)}"
        
        return time.time() - start_time, status

    async def run(self):
        files = [f for f in TEST_DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() not in EXCLUDED_EXTENSIONS]
        files = files[:]  # Limit to 3 files for quick initial benchmark
        print(f"Found {len(files)} files to benchmark: {[f.name for f in files]}")

        methods = [
            # ("Docling", self.benchmark_docling),
            ("PyMuPDF4LLM", self.benchmark_pymupdf4llm),
            # ("Marker", self.benchmark_marker),
            ("Kreuzberg", self.benchmark_kreuzberg),
            # ("RapidOCR (first page)", self.benchmark_rapidocr)
        ]

        summary = {}

        for method_name, func in methods:
            print(f"\nBenchmarking {method_name}...", flush=True)
            total_time = 0
            success_count = 0
            file_results = []

            for file_path in files:
                print(f"  Processing {file_path.name}...", flush=True)
                if asyncio.iscoroutinefunction(func):
                    duration, status = await func(file_path)
                else:
                    duration, status = func(file_path)
                
                if status == "success":
                    total_time += duration
                    success_count += 1
                
                file_results.append({
                    "file": file_path.name,
                    "duration": duration,
                    "status": status
                })

            summary[method_name] = {
                "total_time": total_time,
                "success_count": success_count,
                "avg_time": total_time / success_count if success_count > 0 else 0,
                "throughput_docs_per_sec": success_count / total_time if total_time > 0 else 0,
                "details": file_results
            }

        # Save results
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(summary, f, indent=4)

        print("\n--- Benchmark Summary ---")
        for method, data in summary.items():
            print(f"{method}:")
            print(f"  Throughput: {data['throughput_docs_per_sec']:.2f} docs/sec")
            print(f"  Avg Time: {data['avg_time']:.4f} sec")
            print(f"  Success: {data['success_count']}/{len(files)}")

if __name__ == "__main__":
    benchmark = Benchmark()
    asyncio.run(benchmark.run())
