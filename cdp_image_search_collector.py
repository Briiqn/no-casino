import os
import json
import time
import urllib.parse
import urllib.request
import requests
import websocket
from concurrent.futures import ThreadPoolExecutor

QUERIES = [
    # 1. Specific MrBeast scam memes
    ("mrbeast scam meme", "assets/scam_memes", 40),
    ("mrbeast casino meme", "assets/scam_memes", 40),
    ("mrbeast exit scam meme", "assets/scam_memes", 40),
    ("mrbeast withdrawal failed meme", "assets/scam_memes", 40),
    ("mrbeast crypto scam meme discord", "assets/scam_memes", 40),
    
    # 2. MrBeast logos and variations
    ("mrbeast logo transparent png", "assets/mrbeast_logos", 50),
    ("mrbeast logo variations panther pink lightning", "assets/mrbeast_logos", 40),
    ("mrbeast burger feastables logo png", "assets/mrbeast_logos", 40),
    
    # 3. Jimmy Donaldson faces (pointing, smiling, thumbnails)
    ("jimmy donaldson mrbeast face", "assets/jimmy_faces", 50),
    ("mrbeast youtube thumbnail face pointing smiling", "assets/jimmy_faces", 50),
    
    # 4. Elon Musk tweets and normal X posts
    ("elon musk tweet screenshot x post", "assets/elon_x_screenshots", 50),
    ("elon musk crypto dogecoin tweet screenshot", "assets/elon_x_screenshots", 40),
    ("normal twitter x post screenshot", "assets/elon_x_screenshots", 50),
]

def get_page_tab():
    res = urllib.request.urlopen("http://localhost:9222/json").read()
    tabs = json.loads(res.decode())
    page_tabs = [t for t in tabs if t.get("type") == "page"]
    return page_tabs[0]

def collect_urls_for_query(ws, query, max_count=50):
    url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}&iar=images&iax=images&ia=images"
    ws.send(json.dumps({"id": 100, "method": "Page.navigate", "params": {"url": url}}))
    time.sleep(2.5)
    
    # Scroll down twice to trigger infinite scroll
    for _ in range(2):
        ws.send(json.dumps({
            "id": 101,
            "method": "Runtime.evaluate",
            "params": {"expression": "window.scrollTo(0, document.body.scrollHeight);"}
        }))
        time.sleep(1.0)
        
    js = """
    Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src && src.includes("external-content.duckduckgo.com") && src.includes("ipo=images"))
    """
    ws.send(json.dumps({"id": 102, "method": "Runtime.evaluate", "params": {"expression": js, "returnByValue": True}}))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == 102:
            urls = resp.get("result", {}).get("result", {}).get("value", [])
            return urls[:max_count]

def download_image(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def main():
    tab = get_page_tab()
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"])
    
    all_download_tasks = []
    
    print("Beginning Chromium CDP Search queries...")
    for query, out_dir, count in QUERIES:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Searching: '{query}' -> {out_dir} (target: {count})...")
        urls = collect_urls_for_query(ws, query, max_count=count)
        print(f"  Found {len(urls)} image URLs.")
        clean_prefix = query.replace(" ", "_").replace(":", "")[:20]
        for idx, u in enumerate(urls):
            filename = f"{clean_prefix}_{idx:03d}.jpg"
            out_path = os.path.join(out_dir, filename)
            all_download_tasks.append((u, out_path))
            
    ws.close()
    print(f"\nTotal image download tasks queued: {len(all_download_tasks)}")
    
    # Download concurrently
    def worker(task):
        u, path = task
        return download_image(u, path)
        
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker, all_download_tasks))
        
    succeeded = sum(results)
    print(f"Successfully downloaded {succeeded} images across search categories!")

if __name__ == "__main__":
    main()
