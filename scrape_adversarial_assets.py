import os
import json
import time
import base64
import urllib.request
import websocket
from PIL import Image

def get_page_tab():
    res = urllib.request.urlopen("http://localhost:9222/json").read()
    tabs = json.loads(res.decode())
    page_tabs = [t for t in tabs if t.get("type") == "page"]
    if not page_tabs:
        # Create new tab
        new_tab = json.loads(urllib.request.urlopen("http://localhost:9222/json/new").read().decode())
        return new_tab
    return page_tabs[0]

class CDPBrowser:
    def __init__(self):
        tab = get_page_tab()
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"])
        self.cmd_id = 1
        
    def cmd(self, method, params={}):
        cid = self.cmd_id
        self.cmd_id += 1
        self.ws.send(json.dumps({"id": cid, "method": method, "params": params}))
        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == cid:
                return resp
                
    def navigate(self, url, wait_seconds=3):
        print(f"Navigating to {url}...")
        self.cmd("Page.navigate", {"url": url})
        time.sleep(wait_seconds)
        
    def evaluate(self, js, await_promise=False):
        res = self.cmd("Runtime.evaluate", {
            "expression": js,
            "awaitPromise": await_promise,
            "returnByValue": True
        })
        return res.get("result", {}).get("result", {}).get("value")
        
    def download_image(self, url, out_path):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return True
        js = f"""
        (async () => {{
          try {{
            const resp = await fetch("{url}", {{ mode: 'cors' }});
            if (!resp.ok) return null;
            const blob = await resp.blob();
            return new Promise(resolve => {{
              const reader = new FileReader();
              reader.onloadend = () => resolve(reader.result);
              reader.readAsDataURL(blob);
            }});
          }} catch(e) {{
            return null;
          }}
        }})()
        """
        data_url = self.evaluate(js, await_promise=True)
        if not data_url or "," not in data_url:
            return False
        try:
            _, b64data = data_url.split(",", 1)
            raw = base64.b64decode(b64data)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(raw)
            return True
        except Exception as e:
            print(f"Error saving {out_path}: {e}")
            return False

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass

def main():
    browser = CDPBrowser()
    
    os.makedirs("assets/logos", exist_ok=True)
    os.makedirs("assets/faces", exist_ok=True)
    os.makedirs("assets/memes", exist_ok=True)
    
    # 1. Scrape MrBeast Wikipedia for official logos and photos
    browser.navigate("https://en.wikipedia.org/wiki/MrBeast", wait_seconds=3)
    wiki_images = browser.evaluate("""
        Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src.startsWith("http"))
    """) or []
    
    print(f"Found {len(wiki_images)} images on MrBeast Wikipedia.")
    for idx, url in enumerate(wiki_images):
        if "MrBeast" in url or "Beast" in url or "crew" in url:
            if "logo" in url.lower() or "png" in url.lower():
                out = f"assets/logos/mrbeast_logo_{idx}.png"
            else:
                out = f"assets/faces/mrbeast_face_{idx}.jpg"
            if browser.download_image(url, out):
                print(f"Downloaded: {out}")
                
    # 2. Scrape Elon Musk Wikipedia
    browser.navigate("https://en.wikipedia.org/wiki/Elon_Musk", wait_seconds=3)
    elon_images = browser.evaluate("""
        Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src.startsWith("http"))
    """) or []
    for idx, url in enumerate(elon_images[:15]):
        if "Elon" in url or "Musk" in url or "crop" in url:
            out = f"assets/faces/elon_{idx}.jpg"
            if browser.download_image(url, out):
                print(f"Downloaded: {out}")
                
    # 3. Scrape Twitter / X logo and screenshot
    browser.navigate("https://en.wikipedia.org/wiki/Twitter", wait_seconds=3)
    twit_images = browser.evaluate("""
        Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src.startsWith("http"))
    """) or []
    for idx, url in enumerate(twit_images[:15]):
        if "logo" in url.lower() or "twitter" in url.lower() or "screenshot" in url.lower():
            out = f"assets/logos/twitter_asset_{idx}.png"
            if browser.download_image(url, out):
                print(f"Downloaded: {out}")

    # 4. Scrape Internet Meme Wikipedia for classic memes (Doge, Pepe, Rickroll, etc.)
    browser.navigate("https://en.wikipedia.org/wiki/Internet_meme", wait_seconds=3)
    meme_images = browser.evaluate("""
        Array.from(document.querySelectorAll("img")).map(img => img.src).filter(src => src.startsWith("http"))
    """) or []
    for idx, url in enumerate(meme_images[:20]):
        out = f"assets/memes/meme_{idx}.jpg"
        if browser.download_image(url, out):
            print(f"Downloaded: {out}")
            
    # 5. Direct high-res downloads from Wikimedia Commons APIs
    direct_assets = [
        ("assets/logos/mrbeast_panther_hires.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/MrBeast.png/500px-MrBeast.png"),
        ("assets/faces/jimmy_donaldson_2026.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/MrBeast_in_2026_%28cropped_4%29.png/500px-MrBeast_in_2026_%28cropped_4%29.png"),
        ("assets/faces/jimmy_2023.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/MrBeast_2023_%28cropped%29.jpg/500px-MrBeast_2023_%28cropped%29.jpg"),
        ("assets/logos/x_logo_white.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/X_logo_2023_%28white%29.png/500px-X_logo_2023_%28white%29.png"),
        ("assets/logos/twitter_blue_bird.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Logo_of_Twitter.svg/500px-Logo_of_Twitter.svg.png"),
        ("assets/memes/doge.jpg", "https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Original_Doge_meme.jpg/500px-Original_Doge_meme.jpg"),
        ("assets/memes/nyan_cat.gif", "https://upload.wikimedia.org/wikipedia/en/thumb/e/ed/Nyan_cat_250px_frame.PNG/500px-Nyan_cat_250px_frame.PNG"),
        ("assets/faces/elon_royal.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop1%29.jpg/500px-Elon_Musk_Royal_Society_%28crop1%29.jpg"),
    ]
    
    print("\nFetching direct high-res assets via CDP fetch...")
    for out, url in direct_assets:
        if browser.download_image(url, out):
            print(f"Downloaded: {out}")
            
    browser.close()
    print("\nAsset collection via Chromium CDP complete!")

if __name__ == "__main__":
    main()
