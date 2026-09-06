import urllib.request

# 1. Fetch a real photo
real_url = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=600"
urllib.request.urlretrieve(real_url, "real_sample.jpg")
print("[+] Downloaded real_sample.jpg")

# 2. Fetch a StyleGAN fake face
fake_url = "https://thispersondoesnotexist.com"
# A browser header prevents the site from blocking the script
req = urllib.request.Request(fake_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open("ai_sample.jpg", "wb") as out_file:
    out_file.write(response.read())
print("[+] Downloaded ai_sample.jpg")