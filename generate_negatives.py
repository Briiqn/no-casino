import os
import random
import glob
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

OUTPUT_DIR = "negatives"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Collect local system images
def collect_system_images():
    search_paths = [
        "/usr/share/backgrounds",
        "/usr/share/wallpapers",
        "/usr/share/plasma/wallpapers",
    ]
    imgs = []
    for p in search_paths:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            imgs.extend(glob.glob(f"{p}/**/{ext}", recursive=True))
    
    saved_count = 0
    for idx, path in enumerate(imgs):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((800, 800))
                out_path = os.path.join(OUTPUT_DIR, f"sys_wallpaper_{idx:04d}.jpg")
                im.save(out_path, quality=85)
                saved_count += 1
        except Exception:
            continue
    print(f"Collected {saved_count} system images.")
    return saved_count

# 2. Download diverse web photos from fast CDN (Picsum)
def download_picsum_images(count=350):
    print(f"Downloading {count} varied photos from CDN...")
    downloaded = 0
    def fetch_one(idx):
        url = f"https://picsum.photos/id/{idx % 1000}/500/500"
        out_path = os.path.join(OUTPUT_DIR, f"picsum_{idx:04d}.jpg")
        if os.path.exists(out_path):
            return True
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
            # verify it's a valid image
            with Image.open(out_path) as im:
                im.verify()
            return True
        except Exception:
            if os.path.exists(out_path):
                os.remove(out_path)
            return False

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(fetch_one, range(10, 10 + count)))
    downloaded = sum(results)
    print(f"Downloaded {downloaded} CDN photos.")
    return downloaded

# 3. Simulate camera monitor photograph effect (scanlines, moire, glare, angle)
def apply_monitor_photo_effect(im):
    arr = np.array(im, dtype=np.float32)
    h, w, c = arr.shape
    
    # 1. Subtle scanlines (CRT/LCD subpixel grid)
    scanline_freq = random.randint(3, 6)
    y_idx = np.arange(h)[:, None, None]
    scanlines = 1.0 - 0.08 * (y_idx % scanline_freq == 0)
    arr = arr * scanlines
    
    # 2. Slight moire effect
    if random.random() > 0.5:
        x_idx = np.arange(w)[None, :, None]
        moire = 1.0 + 0.05 * np.sin(x_idx * 0.1 + y_idx * 0.1)
        arr = arr * moire
        
    # 3. Vignetting / camera glare
    y_norm = (np.arange(h) - h / 2) / (h / 2)
    x_norm = (np.arange(w) - w / 2) / (w / 2)
    dist = np.sqrt(y_norm[:, None]**2 + x_norm[None, :]**2)[:, :, None]
    vignette = np.clip(1.0 - 0.25 * (dist**1.8), 0.6, 1.0)
    arr = arr * vignette
    
    # Clip and return PIL Image
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    res_im = Image.fromarray(arr)
    
    # Slight blur / camera focus softening
    if random.random() > 0.5:
        res_im = res_im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    return res_im

# 4. Synthesize realistic social media posts (Legitimate MrBeast, creators, gaming, news)
LEGIT_MRBEAST_TWEETS = [
    "I spent 7 days in an abandoned city surviving on military rations!",
    "We just opened our 100th well in Africa bringing fresh clean drinking water to over 300,000 people!",
    "Feastables chocolate bars are officially hitting store shelves across the UK and Australia this week!",
    "Our new video is finally live! 100 people compete in the world's hardest obstacle course for $500,000.",
    "TeamTrees just planted its 25 millionth tree. Thank you everyone who donated and made this possible!",
    "I survived 50 hours in Antarctica with no heat. Hardest challenge of my entire life.",
    "Giving away 10 Tesla cars to random subscribers who retweet this video clip before midnight!",
    "Beast Games is currently in production with Amazon Prime Video. It's the biggest game show in human history.",
    "Carl and Nolan tried to build a real life LEGO house and live in it for 24 hours.",
    "Just donated 20,000 pairs of shoes to kids in need with Beast Philanthropy."
]

GENERAL_TWEETS = [
    ("Elon Musk", "@elonmusk", "Starship Flight 7 launched successfully and completed full orbital insertion."),
    ("MKBHD", "@MKBHD", "The new smartphone camera sensor is genuinely wild. Full review dropping tomorrow at 9 AM."),
    ("IGN", "@IGN", "Grand Theft Auto VI release date officially confirmed. Here is everything we know so far."),
    ("OpenAI", "@OpenAI", "Introducing our next generation reasoning model with enhanced multimodal perception."),
    ("The Verge", "@verge", "Why modern laptop displays are finally getting OLED panels across all price points."),
    ("NASA", "@NASA", "James Webb Space Telescope detects organic molecules in the earliest galaxy ever observed."),
    ("YouTube", "@YouTube", "Introducing new creator studio tools to help you manage analytics and community posts."),
    ("PC Gamer", "@pcgamer", "The RTX 5080 benchmarks are in: massive gains in rasterization and path tracing performance.")
]

def synthesize_social_media_posts(count=800):
    print(f"Generating {count} synthetic social media & UI screenshots...")
    font = ImageFont.load_default()
    
    for i in range(count):
        is_dark_mode = random.random() > 0.3
        bg_color = (21, 32, 43) if is_dark_mode else (245, 248, 250)
        card_bg = (0, 0, 0) if is_dark_mode else (255, 255, 255)
        text_primary = (255, 255, 255) if is_dark_mode else (15, 20, 25)
        text_secondary = (136, 153, 166) if is_dark_mode else (83, 100, 113)
        accent_color = (29, 155, 240) # Twitter blue
        
        w, h = random.randint(700, 1000), random.randint(700, 1000)
        im = Image.new("RGB", (w, h), color=bg_color)
        draw = ImageDraw.Draw(im)
        
        # Draw simulated browser tab bar or phone app header
        draw.rectangle([0, 0, w, 40], fill=(30, 30, 35) if is_dark_mode else (230, 230, 235))
        draw.text((20, 12), "x.com / Home", fill=text_secondary, font=font)
        
        # Draw tweet card
        card_box = [30, 60, w - 30, h - 60]
        draw.rectangle(card_box, fill=card_bg, outline=text_secondary, width=1)
        
        # Decide if this is a legitimate MrBeast tweet or another account
        if i % 3 == 0:
            author_name = "MrBeast"
            handle = "@MrBeast"
            body_text = random.choice(LEGIT_MRBEAST_TWEETS)
            avatar_color = (0, 180, 216)
        else:
            author_name, handle, body_text = random.choice(GENERAL_TWEETS)
            avatar_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            
        # Draw avatar
        draw.ellipse([50, 80, 110, 140], fill=avatar_color)
        draw.text((70, 105), author_name[0], fill=(255, 255, 255), font=font)
        
        # Draw author info
        draw.text((125, 90), f"{author_name}", fill=text_primary, font=font)
        draw.ellipse([125 + len(author_name)*8, 92, 137 + len(author_name)*8, 104], fill=accent_color) # checkmark circle
        draw.text((150 + len(author_name)*8, 90), handle, fill=text_secondary, font=font)
        
        # Draw tweet body
        words = body_text.split()
        lines = []
        cur_line = []
        for word in words:
            if len(" ".join(cur_line + [word])) > 45:
                lines.append(" ".join(cur_line))
                cur_line = [word]
            else:
                cur_line.append(word)
        if cur_line:
            lines.append(" ".join(cur_line))
            
        y_text = 150
        for line in lines:
            draw.text((125, y_text), line, fill=text_primary, font=font)
            y_text += 22
            
        # Draw simulated media attachment or video player
        if random.random() > 0.4:
            media_h = random.randint(180, 300)
            draw.rectangle([125, y_text + 15, w - 60, y_text + 15 + media_h], fill=(random.randint(40, 80), random.randint(40, 80), random.randint(60, 100)))
            draw.rectangle([125, y_text + 15 + media_h - 25, w - 60, y_text + 15 + media_h], fill=(0, 0, 0, 128))
            draw.text((140, y_text + 15 + media_h - 20), "HD 1080p Video • 12:45", fill=(255, 255, 255), font=font)
            y_text += media_h + 30
            
        # Draw interactions bar
        draw.text((125, y_text + 20), "💬 12.4K     🔄 45.8K     ❤️ 389K     📊 14.2M", fill=text_secondary, font=font)
        
        # Apply camera/monitor effect to half the synthetic images
        if random.random() > 0.4:
            im = apply_monitor_photo_effect(im)
            
        out_path = os.path.join(OUTPUT_DIR, f"synth_social_{i:04d}.jpg")
        im.save(out_path, quality=85)
    print(f"Generated {count} synthetic social media images.")

# 5. Synthesize general web UI & desktop screenshots (Wikipedia, Code, News)
def synthesize_web_desktop_screenshots(count=400):
    print(f"Generating {count} web UI & desktop screenshots...")
    font = ImageFont.load_default()
    
    pages = [
        ("Wikipedia - Artificial Intelligence", ["Artificial intelligence (AI) is the intelligence of machines or software.", "Major AI research goals include reasoning, knowledge representation,", "planning, natural language processing, and machine perception."]),
        ("GitHub - Repository Overview", ["commit 8f4a12c: Update training pipeline and model architecture", "142 branches | 29 releases | 4.8k stars", "README.md: Fast, lightweight image classification in PyTorch."]),
        ("Tech News Daily", ["Global Semiconductor Market Reaches Record High in Q3", "Analysts report an unprecedented surge in demand for AI accelerators", "and energy-efficient enterprise server architectures."]),
        ("Online Store - Electronics", ["Mechanical Keyboard RGB Backlit - $89.99 [In Stock]", "Wireless Ergonomic Mouse - $49.99 [Free Shipping]", "34-inch Ultrawide Curved Gaming Monitor - $399.99"])
    ]
    
    for i in range(count):
        title, content_lines = random.choice(pages)
        w, h = random.randint(800, 1100), random.randint(700, 950)
        im = Image.new("RGB", (w, h), color=(240, 242, 245))
        draw = ImageDraw.Draw(im)
        
        # Browser chrome
        draw.rectangle([0, 0, w, 50], fill=(220, 224, 230))
        draw.ellipse([15, 18, 27, 30], fill=(255, 95, 87)) # red
        draw.ellipse([35, 18, 47, 30], fill=(255, 189, 46)) # yellow
        draw.ellipse([55, 18, 67, 30], fill=(39, 201, 63)) # green
        draw.rectangle([90, 10, w - 100, 40], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((110, 18), f"https://www.{title.split()[0].lower()}.org", fill=(80, 80, 80), font=font)
        
        # Content body
        draw.rectangle([40, 80, w - 40, h - 40], fill=(255, 255, 255), outline=(210, 210, 210))
        draw.text((70, 110), title, fill=(20, 20, 20), font=font)
        draw.line([70, 135, w - 70, 135], fill=(200, 200, 200), width=1)
        
        y = 160
        for line in content_lines:
            draw.text((70, y), line, fill=(50, 50, 50), font=font)
            y += 30
            
        # Draw some colored diagrams/boxes
        draw.rectangle([70, y + 20, 320, y + 160], fill=(235, 240, 250), outline=(150, 180, 220))
        draw.text((90, y + 80), "Figure 1: Architecture Overview", fill=(70, 100, 150), font=font)
        
        if random.random() > 0.4:
            im = apply_monitor_photo_effect(im)
            
        out_path = os.path.join(OUTPUT_DIR, f"synth_web_{i:04d}.jpg")
        im.save(out_path, quality=85)
    print(f"Generated {count} web UI screenshots.")

def main():
    n_sys = collect_system_images()
    n_picsum = download_picsum_images(count=500)
    synthesize_social_media_posts(count=1500)
    synthesize_web_desktop_screenshots(count=800)
    
    total = len(glob.glob(f"{OUTPUT_DIR}/*.jpg"))
    print(f"\nTotal negative images compiled: {total}")

if __name__ == "__main__":
    main()
