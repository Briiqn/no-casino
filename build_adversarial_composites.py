import os
import glob
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

OUTPUT_DIR = "adversarial_composites"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_rgba_overlay(path):
    try:
        im = Image.open(path)
        if im.mode != "RGBA":
            im = im.convert("RGBA")
            data = np.array(im)
            r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
            # Make pure white or black corners transparent if needed
            white_mask = (r > 240) & (g > 240) & (b > 240)
            data[white_mask, 3] = 0
            im = Image.fromarray(data)
        return im
    except Exception:
        return None

def build_phone_mockup_overlay(logo_im, w=220, h=400):
    phone = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(phone)
    draw.rounded_rectangle([0, 0, w, h], radius=24, fill=(15, 15, 18), outline=(60, 60, 70), width=4)
    screen_box = [8, 8, w - 8, h - 8]
    draw.rounded_rectangle(screen_box, radius=18, fill=(135, 206, 235, 255))
    
    if logo_im:
        lw = int(w * 0.75)
        lh = int(lw * (logo_im.height / max(1, logo_im.width)))
        resized_logo = logo_im.resize((lw, lh), Image.Resampling.LANCZOS)
        lx = (w - lw) // 2
        ly = (h - lh) // 2
        phone.paste(resized_logo, (lx, ly), resized_logo if resized_logo.mode == "RGBA" else None)
        
    draw.rounded_rectangle([(w//2) - 25, 12, (w//2) + 25, 26], radius=7, fill=(0, 0, 0, 255))
    return phone

def slap_overlay_on_base(base_im, overlay_im, mode="random"):
    bw, bh = base_im.size
    ow, oh = overlay_im.size
    
    if mode == "phone_mockup":
        target_h = int(bh * random.uniform(0.5, 0.8))
        target_w = int(target_h * (ow / max(1, oh)))
        resized_overlay = overlay_im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        x = random.randint(int(bw * 0.3), max(int(bw * 0.3), bw - target_w - 10))
        y = random.randint(max(0, bh - target_h - 20), bh - target_h)
    else:
        target_w = random.randint(int(bw * 0.15), int(bw * 0.45))
        target_h = int(target_w * (oh / max(1, ow)))
        resized_overlay = overlay_im.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        if random.random() > 0.5:
            resized_overlay = resized_overlay.rotate(random.randint(-25, 25), expand=True, resample=Image.Resampling.BICUBIC)
            target_w, target_h = resized_overlay.size
            
        x = random.randint(10, max(10, bw - target_w - 10))
        y = random.randint(10, max(10, bh - target_h - 10))
        
    base_rgba = base_im.convert("RGBA")
    base_rgba.paste(resized_overlay, (x, y), resized_overlay if resized_overlay.mode == "RGBA" else None)
    return base_rgba.convert("RGB")

def main():
    random.seed(42)
    print("Loading base images and overlays...")
    
    # Base images pool: adversarial folder + elon/x screenshots + negatives
    adv_files = glob.glob("adversarial/*")
    base_files = adv_files + glob.glob("assets/elon_x_screenshots/*") + glob.glob("negatives/*.jpg")
    print(f"Total base images pool: {len(base_files)}")
    
    # Overlays: logos and faces
    logo_paths = glob.glob("assets/mrbeast_logos/*") + glob.glob("assets/logos/*")
    face_paths = glob.glob("assets/jimmy_faces/*") + glob.glob("assets/faces/*")
    elon_paths = glob.glob("assets/elon_x_screenshots/*")
    
    logos = [load_rgba_overlay(p) for p in logo_paths[:100]]
    logos = [l for l in logos if l is not None]
    
    faces = [load_rgba_overlay(p) for p in face_paths[:80]]
    faces = [f for f in faces if f is not None]
    
    elons = [load_rgba_overlay(p) for p in elon_paths[:80]]
    elons = [e for e in elons if e is not None]
    
    print(f"Loaded {len(logos)} logos, {len(faces)} Jimmy faces, {len(elons)} Elon/X assets.")
    
    phone_mockups = [build_phone_mockup_overlay(l) for l in logos[:20]]
    
    count = 3500
    print(f"Generating {count} adversarial composite images...")
    for i in range(count):
        base_path = random.choice(base_files)
        try:
            with Image.open(base_path) as im:
                im = im.convert("RGB")
        except Exception:
            continue
            
        dice = random.random()
        if dice < 0.30 and phone_mockups:
            overlay = random.choice(phone_mockups)
            res = slap_overlay_on_base(im, overlay, mode="phone_mockup")
        elif dice < 0.65 and logos:
            overlay = random.choice(logos)
            res = slap_overlay_on_base(im, overlay, mode="random")
        elif dice < 0.85 and faces:
            overlay = random.choice(faces)
            res = slap_overlay_on_base(im, overlay, mode="random")
        elif elons:
            overlay = random.choice(elons)
            res = slap_overlay_on_base(im, overlay, mode="random")
        else:
            continue
            
        out_path = os.path.join(OUTPUT_DIR, f"adv_composite_{i:04d}.jpg")
        res.save(out_path, quality=85)
        
    total_composites = len(glob.glob(f"{OUTPUT_DIR}/*.jpg"))
    print(f"Generated {total_composites} composites successfully in {OUTPUT_DIR}!")

if __name__ == "__main__":
    main()
