import asyncio
import random
import time
import statistics
from typing import Dict
import httpx

# Sesuaikan URL ini dengan port dan host API FastAPI Anda
API_URL = "http://localhost:80/predict" 
REQUESTS_PER_SECOND = 10
DURATION_SECONDS = 60

def generate_payload() -> Dict[str, float]:
    """Generate data dummy acak yang masuk akal untuk fitur air quality."""
    return {
        "pm2_5": random.uniform(0.0, 100.0),
        "pm10": random.uniform(0.0, 150.0),
        "co": random.uniform(0.0, 5.0),
        "no2": random.uniform(0.0, 100.0),
        "o3": random.uniform(0.0, 100.0),
        "so2": random.uniform(0.0, 50.0),
    }

async def send_request(client: httpx.AsyncClient, request_id: int) -> Dict:
    """Mengirim 1 request POST dan mengukur latency."""
    start_time = time.perf_counter()
    payload = generate_payload()
    try:
        # Timeout 5 detik per request
        response = await client.post(API_URL, json=payload, timeout=5.0)
        latency = (time.perf_counter() - start_time) * 1000  # konversi ke ms
        return {
            "id": request_id,
            "status": response.status_code,
            "latency_ms": latency,
            "success": response.is_success
        }
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        return {
            "id": request_id,
            "status": 0,
            "latency_ms": latency,
            "success": False,
            "error": str(e)
        }

async def main():
    print(f"Memulai load test: {REQUESTS_PER_SECOND} req/detik selama {DURATION_SECONDS} detik...")
    print(f"Target URL: {API_URL}\n")
    
    total_requests = 0
    success_count = 0
    fail_count = 0
    latencies = []
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        for sec in range(DURATION_SECONDS):
            sec_start = time.perf_counter()
            
            tasks = []
            # Siapkan 10 task untuk detik ini
            for _ in range(REQUESTS_PER_SECOND):
                total_requests += 1
                tasks.append(send_request(client, total_requests))
            
            # Kirim 10 request secara konkuren (bersamaan)
            results = await asyncio.gather(*tasks)
            
            # Kumpulkan hasil
            for res in results:
                if res["success"]:
                    success_count += 1
                else:
                    fail_count += 1
                latencies.append(res["latency_ms"])
                
            # Hitung sisa waktu untuk tidur agar total waktu per siklus tepat 1 detik
            elapsed = time.perf_counter() - sec_start
            sleep_time = 1.0 - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
            # Cetak progress setiap 10 detik
            if (sec + 1) % 10 == 0:
                print(f"Progress: {sec + 1}/{DURATION_SECONDS} detik | Sukses: {success_count} | Gagal: {fail_count}")

    total_time = time.time() - start_time
    
    print("\n=== Ringkasan Load Test ===")
    print(f"Total Waktu   : {total_time:.2f} detik")
    print(f"Total Request : {total_requests}")
    print(f"Request Sukses: {success_count}")
    print(f"Request Gagal : {fail_count}")
    
    if latencies:
        print(f"\n--- Statistik Latency ---")
        print(f"Rata-rata     : {statistics.mean(latencies):.2f} ms")
        print(f"Min           : {min(latencies):.2f} ms")
        print(f"Max           : {max(latencies):.2f} ms")
        
        # Hitung P95 dan P99
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]
        print(f"P95 (95%)     : {p95:.2f} ms")
        print(f"P99 (99%)     : {p99:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())