import os
import json
import time
import urllib.parse
import urllib.request
import requests
import websocket
from concurrent.futures import ThreadPoolExecutor

BRAVE_QUERIES = [
    # 1. Specific MrBeast scam & exit scam memes
    ("mrbeast scam meme", "assets/scam_memes", 50),
    ("mrbeast casino meme", "assets/scam_memes", 50),
    ("mrbeast withdrawal failed exit scam meme", "assets/scam_memes", 50),
    ("mrbeast meme discord funny", "assets/scam_memes", 50),
    
    # 2. MrBeast logos & wallpapers
    ("mrbeast logo png transparent", "assets/mrbeast_logos", 50),
    ("mrbeast logo wallpaper panther", "assets/mrbeast_logos", 50),
    
    # 3. Jimmy Donaldson faces & thumbnails
    ("jimmy donaldson face mrbeast", "assets/jimmy_faces", 50),
    ("mrbeast youtube thumbnail pointing", "assets/jimmy_faces", 50),
    
    # 4. Elon Musk tweets & Twitter/X screenshots
    ("elon musk tweet screenshot x post", "assets/elon_x_screenshots", 50),
    ("elon musk crypto tweet screenshot", "assets/elon_x_screenshots", 50),
    ("normal twitter screenshot x post feed", "assets/elon_x_screenshots", 50),
]

def get_page_tab():
    res = urllib.request.urlopen("http://localhost:9222/json").read()
    tabs = json.loads(res.decode())
    page_tabs = [t for t in tabs if t.get("type") == "page"]
    return page_tabs[0]

def collect_brave_urls(ws, query, max_count=50):
    url = f"https://search.brave.com/images?q={urllib.parse.quote(query)}"
    ws.send(json.dumps({"id": 200, "method": "Page.navigate", "params": {"url": url}}))
    time.sleep(3.0)
    
    # Scroll to load more
    for _ in range(2):
        ws.send(json.dumps({
            "id": 201,
            "method": "Runtime.evaluate",
            "params": {"expression": "window.scrollTo(0, document.body.scrollHeight);"}
        }))
        time.sleep(1.0)
        
    js = """
    Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src && src.includes("imgs.search.brave.com"))
    """
    ws.send(json.dumps({"id": 202, "method": "Runtime.evaluate", "params": {"expression": js, "returnByValue": True}}))
    while True:
        resp = json.loads(ws.recv())
        if resp.get("id") == 202:
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
    
    download_tasks = []
    print("Beginning Brave Image Search queries via Chromium CDP...")
    
    for query, out_dir, count in BRAVE_QUERIES:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Brave Search: '{query}' -> {out_dir} (target: {count})...")
        urls = collect_brave_urls(ws, query, max_count=count)
        print(f"  Found {len(urls)} Brave image URLs.")
        clean_prefix = query.replace(" ", "_").replace(":", "")[:18]
        for idx, u in enumerate(urls):
            filename = f"brave_{clean_prefix}_{idx:03d}.jpg"
            out_path = os.path.join(out_dir, filename)
            download_tasks.append((u, out_path))
            
    ws.close()
    print(f"\nTotal Brave download tasks queued: {len(download_tasks)}")
    
    def worker(task):
        u, path = task
        return download_image(u, path)
        
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker, download_tasks))
        
    succeeded = sum(results)
    print(f"Successfully downloaded {succeeded} images from Brave Search!")

if __name__ == "__main__":
    main()
